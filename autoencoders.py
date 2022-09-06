import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0,"taming_transformers")
import taming
import utils

class Transformer(torch.nn.Module):
    
    def __init__(self, weights, resolution=256, download=False):
        super().__init__()
        
        if weights == "imagenet":
            weights = "./weights/2021-04-03T19-39-50_cin_transformer.pth"
            url = "https://github.com/ieee8023/latentshift/releases/download/weights/2021-04-03T19-39-50_cin_transformer.pth"
        elif weights == "faceshq":
            weights = "./weights/2020-11-13T21-41-45_faceshq_transformer.pth"
            url = "https://github.com/ieee8023/latentshift/releases/download/weights/2020-11-13T21-41-45_faceshq_transformer.pth"
        else:
            raise Exception("No weights specified")
        
        if not os.path.isfile(weights):
            if download:
                utils.download(url, weights)
            else:
                print("No weights found, specify download=True to download them.")
        
        try:
            self.model = torch.load(weights)
        except:
            raise Exception(f'Error loading weights, try deleting them and redownloading: rm {weights}')
        
        self.upsample = torch.nn.Upsample(size=(resolution, resolution), mode='bilinear', align_corners=False)
    
    def encode(self, x):
        x = (x*2 - 1.0)
        x = self.upsample(x)
        return self.model.encode(x)[0]
    
    def decode(self, z, image_shape=None):
        xp = self.model.decode(z)
        xp = (xp+1)/2
        xp = torch.clip(xp, 0,1)
        return xp
    
    def forward(self, x):
        return self.decode(self.encode(x))

