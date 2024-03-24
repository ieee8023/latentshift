import sys
import os
import torch
from . import utils
import taming.models.vqgan
import omegaconf

baseurl = "https://github.com/ieee8023/latentshift/releases/download/weights/"
weights_path = utils.get_cache_folder()


class VQGAN(torch.nn.Module):
    """These transformers are based on the first stage of the models
    in the following work:

    Taming Transformers for High-Resolution Image Synthesis
    Patrick Esser and Robin Rombach and Björn Ommer
    https://github.com/CompVis/taming-transformers
    https://arxiv.org/abs/2012.09841
    """
    
    def __init__(self, weights, config=None, download=True):
        super().__init__()
        
        is_gumbel = False
        if weights == "imagenet":
            weights = "vqgan_imagenet_f16_1024.ckpt"
            config = "vqgan_imagenet_f16_1024.yaml"
        elif weights == "faceshq":
            weights = "2020-11-13T21-41-45_faceshq.pth"
            config = "2020-11-13T21-41-45_faceshq.yaml"
        elif weights == "gumbel_f8":
            weights = "vqgan_gumbel_f8.ckpt"
            config = "vqgan_gumbel_f8.yaml"
            is_gumbel=True
        
        if (not os.path.isfile(weights_path + weights)) or (not os.path.isfile(weights_path + config)):
            if download:
                utils.download(baseurl + weights, weights_path + weights)
                utils.download(baseurl + config, weights_path + config)
            else:
                print("No weights found, specify download=True to download them.")
        
        try: 
            c = omegaconf.OmegaConf.load(weights_path + config)
            self.config = c['model']['params']
            if is_gumbel:
                self.model = taming.models.vqgan.GumbelVQ(**self.config)
            else:
                self.model = taming.models.vqgan.VQModel(**self.config)
        except:
            raise Exception(f'Error creating model. Try deleting the config and redownloading if: rm {weights_path + config}')
            
        try:
            a = torch.load(weights_path + weights, map_location=torch.device('cpu'))
            if 'state_dict' in a:
                a = a['state_dict']
            self.model.load_state_dict(a, strict=False);
        except:
            raise Exception(f'Error loading weights, try deleting them and redownloading: rm {weights_path + weights}')
        
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
