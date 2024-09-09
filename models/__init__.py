import logging
from models.classifier import Classifier, NormalizedClassifier
from models.img_resnet import ResNet50, GEN
from models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50
from models.PM_img import PM
from models.Fusion import Fusion


__factory = {
    'resnet50': ResNet50,
    'c2dres50': C2DResNet50,
    'i3dres50': I3DResNet50,
    'ap3dres50': AP3DResNet50,
    'nlres50': NLResNet50,
    'ap3dnlres50': AP3DNLResNet50,
}


def build_model(config, num_identities, num_clothes):
    logger = logging.getLogger('reid.model')
    # Build backbone
    logger.info("Initializing model: {}".format(config.MODEL.NAME))

    logger.info("Init model: '{}'".format(config.MODEL.NAME))
    model = __factory[config.MODEL.NAME](config)
    model2 = PM(feature_dim=config.MODEL.FEATURE_DIM, config=config)
    fusion = Fusion(feature_dim=config.MODEL.FEATURE_DIM)
    
    logger.info("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    logger.info("Model2 size: {:.5f}M".format(sum(p.numel() for p in model2.parameters())/1000000.0))
    logger.info("Fusion size: {:.5f}M".format(sum(p.numel() for p in fusion.parameters())/1000000.0))

    # Build classifier
    if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
        identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
    else:
        identity_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)

    clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)
    
    #classifier of new model

    clothes_classifier2 = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)
    return model, model2, fusion, identity_classifier, clothes_classifier, clothes_classifier2