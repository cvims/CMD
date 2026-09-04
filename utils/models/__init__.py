import logging
from typing import Any, Dict
from torchvision import transforms
import timm

logger = logging.getLogger(__name__)

def _tinyimagenet_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25),
    ])

    test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    return train, test


def _build_tinyimagenet_model(model_name, features_nodes=None):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=200,
    )

    train_transforms, test_transforms = _tinyimagenet_transforms()

    return {
        "model": model,
        "features_nodes": features_nodes or {
            "flatten": "features",
            "head": "logits",
        },
        "input_dim": (3, 224, 224),
        "train_transforms": train_transforms,
        "test_transforms": test_transforms,
    }


def DenseNet121TinyImageNet(features_nodes=None):
    return _build_tinyimagenet_model(
        "densenet121",
        features_nodes,
    )


def WideResNetTinyImageNet(features_nodes=None):
    return _build_tinyimagenet_model(
        "wide_resnet50_2",
        features_nodes,
    )


def ResNet34TinyImageNet(features_nodes=None):
    return _build_tinyimagenet_model(
        "resnet34",
        features_nodes,
    )


def SwinTinyImageNet(features_nodes=None):
    return _build_tinyimagenet_model(
        "swin_tiny_patch4_window7_224",
        features_nodes,
    )


def FastTinyImageNet(features_nodes=None):
    return _build_tinyimagenet_model(
        "fastvit_t8.apple_in1k",
        features_nodes,
    )


def ViTTinyImageNet(features_nodes=None):
    return _build_tinyimagenet_model(
        "vit_tiny_patch16_224",
        features_nodes,
    )


models_registry = {
    # CNN
    "densenet121_tinyimagenet": DenseNet121TinyImageNet,
    "wideresnet_tinyimagenet": WideResNetTinyImageNet,
    "resnet34_tinyimagenet": ResNet34TinyImageNet,
    # Vision Transformers
    "swin_tinyimagenet": SwinTinyImageNet,
    "vit_tinyimagenet": ViTTinyImageNet,
    "fastvit_tinyimagenet": FastTinyImageNet,
}


def get_model_essentials(model_name, features_nodes=None) -> Dict[str, Any]:
    if model_name not in models_registry:
        raise ValueError("Unknown model name: {}".format(model_name))
    return models_registry[model_name](features_nodes=features_nodes)




