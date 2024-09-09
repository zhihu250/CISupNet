from main_bs import parse_option
import torch
from models.PM_img import PM
from models.img_resnet import ResNet50


if __name__=='__main__':
    from torchsummary import summary
    config = parse_option()
    # model2 = PM(feature_dim=config.MODEL.FEATURE_DIM, config=config)
    model2 = ResNet50(config)
    model2 = model2.cuda()
    summary(model2, (3,384,192))