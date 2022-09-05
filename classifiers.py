import sys, os
import torch
import torch.nn.functional as F



class ResNet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        self.model = self.model.eval()
        
    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)


    
class FaceAttribute(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        from attribute_classifier import BranchedTiny
        self.model = BranchedTiny.BranchedTiny("weights/BranchedTiny.ckpt")
        self.model = self.model.eval()
        
        self.targets = self.model.attributes
        
    def forward(self, x):
        return self.model(x)