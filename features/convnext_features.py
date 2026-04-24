import torch
import torch.nn as nn
from torchvision import models

def replace_convlayers_convnext(model, threshold):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_convlayers_convnext(module, threshold)
        if isinstance(module, nn.Conv2d):
            if module.stride[0] == 2:
                if module.in_channels > threshold: #replace bigger strides to reduce receptive field, skip some 2x2 layers. >100 gives output size (26, 26). >300 gives (13, 13)
                    module.stride = tuple(s//2 for s in module.stride)
                    
    return model

def convnext_tiny_26_features(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()    
        model = replace_convlayers_convnext(model, 100) 
    
    return model

def convnext_tiny_13_features(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()    
        model = replace_convlayers_convnext(model, 300) 
    
    return model

class ConvNextMultiStage(nn.Module):
    def __init__(self, pretrained=True, threshold=300):
        super().__init__()

        base = models.convnext_tiny(
            pretrained=pretrained,
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )

        with torch.no_grad():
            base.avgpool = nn.Identity()
            base.classifier = nn.Identity()
            base = replace_convlayers_convnext(base, threshold)

        # IMPORTANT: use same attribute name as torchvision ConvNeXt
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier

    def forward(self, x, return_stage=None):
        penultimate = None

        for i, stage in enumerate(self.features):
            x = stage(x)

            if i == len(self.features) - 3:
                penultimate = x

        final = x

        if return_stage == "penultimate":
            return penultimate

        if return_stage == "both":
            return {
                "penultimate": penultimate,
                "final": final
            }

        return final