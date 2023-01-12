import sys
import os
import torch
import utils
import taming.models.vqgan
import omegaconf


class VQGAN(torch.nn.Module):
    """These transformers are based on the first stage of the models
    in the following work:

    Taming Transformers for High-Resolution Image Synthesis
    Patrick Esser and Robin Rombach and Björn Ommer
    https://github.com/CompVis/taming-transformers
    https://arxiv.org/abs/2012.09841
    """
    
    def __init__(self, weights, download=True):
        super().__init__()
        
        if weights == "imagenet":
            weights = "./weights/vqgan_imagenet_f16_1024.ckpt"
            config = "./weights/vqgan_imagenet_f16_1024.yaml"
            url = "https://github.com/ieee8023/latentshift/releases/download/weights/vqgan_imagenet_f16_1024.ckpt"
        elif weights == "faceshq":
            weights = "./weights/2020-11-13T21-41-45_faceshq.pth"
            config = "./weights/2020-11-13T21-41-45_faceshq.yaml"
            url = "https://github.com/ieee8023/latentshift/releases/download/weights/2020-11-13T21-41-45_faceshq.pth"
        else:
            raise Exception("No weights specified")
        
        if not os.path.isfile(weights):
            if download:
                utils.download(url, weights)
            else:
                print("No weights found, specify download=True to download them.")
        
        try: 
            c = omegaconf.OmegaConf.load(config)
            self.config = c['model']['params']
            self.model = taming.models.vqgan.VQModel(**self.config)
        except:
            raise Exception(f'Error creating model.')
            
        try:
            a = torch.load(weights, map_location=torch.device('cpu'))
            if 'state_dict' in a:
                a = a['state_dict']
            self.model.load_state_dict(a, strict=False);
        except:
            raise Exception(f'Error loading weights, try deleting them and redownloading: rm {weights}')
        
        self.resolution = self.config['ddconfig']['resolution']
        self.upsample = torch.nn.Upsample(size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
    
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
