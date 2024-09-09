import time
import datetime
import logging
import torch
from apex import amp
from tools.utils import AverageMeter

# 正常训练一个resnet50以及一个classifier
def train_cal(config, epoch, model, model2, fuse, classifier, clothes_classifier, clothes_classifier2, criterion_cla, criterion_pair, 
    criterion_clothes, criterion_adv, optimizer, optimizer2, optimizer_cc, trainloader, pid2clothes, kl, recon_cloth):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    batch_clothes_loss2 = AverageMeter()
    batch_loss2 = AverageMeter()
    batch_kl_loss = AverageMeter()
    corrects = AverageMeter()
    corrects2 = AverageMeter()
    corrects3 = AverageMeter()
    clothes_corrects = AverageMeter()
    clothes_corrects2 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    batch_rec_loss = AverageMeter()

    model.train()
    model2.train()
    fuse.train()
    classifier.train()
    clothes_classifier.train()
    clothes_classifier2.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids, cloth, _, _) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        
        pids = pids.cpu()
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        print(imgs.shape)
        cloth = cloth.cuda()
        
        # Measure data loading timeq
        data_time.update(time.time() - end)
        # Forward
        pri_feat, features = model(imgs) # torch.size([32,4096])
        
        # 衣服
        pri_feat2, features2, cloth_img = model2(imgs)

        pri_feat2 = pri_feat2.clone().detach()
        features_fuse = fuse(pri_feat, pri_feat2)
        # ID
        outputs = classifier(features)
        
        outputs2 = clothes_classifier2(features2)
        
        # clothes score on id classifier
        outputs3 = classifier(features_fuse) 

        # new_pred_clothes2 = clothes_classifier2(features2)
        # loss2 = criterion_adv(new_pred_clothes2, clothes_ids, pos_mask)

        pred_clothes = clothes_classifier(features.detach()) # no grad

        _, preds = torch.max(outputs.data, 1) # return (max_value, index), 1 indicates dim=1
    
        _, preds3 = torch.max(outputs3.data, 1)
        
        # Update the clothes discriminator
        clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()
            if config.TRAIN.AMP:
                with amp.scale_loss(clothes_loss, optimizer_cc) as scaled_loss:
                    scaled_loss.backward()
            else:
                clothes_loss.backward()
            optimizer_cc.step()

        # Update the backbone
        new_pred_clothes = clothes_classifier(features)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)
        
        # 衣服
        _, pred_clothes2 = torch.max(outputs2.data, 1)
        # outputs2_no_grad = clothes_classifier2(features2.detach())

        Q = new_pred_clothes.clone().detach()
        P = outputs2.clone()
        Q = torch.nn.functional.softmax(Q, dim=-1)
        P = torch.nn.functional.softmax(P, dim=-1)

        # Update the clothes discriminator 2

        clothes_loss2 = criterion_clothes(outputs2, clothes_ids)

        kl_loss = kl(torch.log(Q), P, reduction='sum') + kl(torch.log(P), Q, reduction='sum')
        # cloth_loss = recon_cloth(cloth_img, cloth)

        if epoch >= config.TRAIN.START_EPOCH_CC:
            # loss2 = clothes_loss2 + config.k_kl * kl_loss + config.PARA.RECON_RATIO * cloth_loss
            loss2 = clothes_loss2 + config.k_kl * kl_loss
        else:
            # loss2 = clothes_loss2 + config.PARA.RECON_RATIO * cloth_loss
            loss2 = clothes_loss2

        optimizer2.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss2, optimizer2) as scaled_loss2:
                scaled_loss2.backward()
        else:
            loss2.backward()
        optimizer2.step()

        GENERAL_EPOCH = config.TRAIN.START_EPOCH_ADV

        # Compute loss
        if epoch >= GENERAL_EPOCH:
            cla_loss = criterion_cla(outputs, pids) + config.k_cal * criterion_cla( outputs3-outputs, pids)
        else:
            cla_loss = criterion_cla(outputs, pids)
            
        pair_loss = criterion_pair(features, pids)
        adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)

        if epoch >= config.TRAIN.START_EPOCH_ADV:
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss 
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss 
        
        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        corrects2.update(torch.sum(pred_clothes2 == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        corrects3.update(torch.sum(preds3 == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        clothes_corrects2.update(torch.sum(pred_clothes2 == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        batch_loss2.update(loss2.item(), clothes_ids.size(0))
        batch_clothes_loss2.update(clothes_loss2.item(), clothes_ids.size(0))
        batch_kl_loss.update(kl_loss.item(), clothes_ids.size(0))
        # batch_rec_loss.update(cloth_loss.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'PairLoss:{pair_loss.avg:.4f} '
                  'CloLoss:{clo_loss.avg:.4f} '
                  'AdvLoss:{adv_loss.avg:.4f} '
                  'clothes_loss2:{clothes_loss2.avg:.4f} '
                  'loss2:{loss2.avg:.4f} '
                  'kl_loss:{kl_loss.avg:.4f} '
                  'Acc:{acc.avg:.2%} '
                  'Acc2:{acc2.avg:.2%} '
                  'Acc3:{acc3.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '
                  'Clo2Acc:{clo2_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                   clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, 
                   clothes_loss2=batch_clothes_loss2, 
                   loss2=batch_loss2, kl_loss=batch_kl_loss,
                   acc=corrects, acc2=corrects2, acc3=corrects3, 
                   clo_acc=clothes_corrects, clo2_acc=clothes_corrects2))
    
    
def train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, 
    criterion_adv, optimizer, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)

        if epoch >= config.TRAIN.START_EPOCH_ADV:
            adv_loss = criterion_adv(features, clothes_ids, pos_mask)
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss  

        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        if epoch >= config.TRAIN.START_EPOCH_ADV: 
            batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
                epoch+1, batch_time=batch_time, data_time=data_time, 
                cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                adv_loss=batch_adv_loss, acc=corrects))
