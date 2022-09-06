import sys, os
import utils
import torch
import torch.nn.functional as F
import pathlib
from attribute_classifier import BranchedTiny


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
        super().__init__(download=False)

        
        filename = "./weights/BranchedTiny.ckpt"
        url = "https://github.com/ieee8023/latentshift/releases/download/weights/BranchedTiny.ckpt"
        
        if not os.path.isfile(filename):
            if download:
                utils.download(url, filename)
            else:
                print("No weights found, specify download=True to download them.")
        
        self.model = BranchedTiny.BranchedTiny(filename)
        self.model = self.model.eval()
        
        self.targets = self.model.attributes
        
    def forward(self, x):
        return self.model(x)