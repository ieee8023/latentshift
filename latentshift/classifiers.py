import os
from . import utils
import torch
import torchvision
from .attribute_classifier import BranchedTiny
from torchvision.models import resnet50, ResNet50_Weights

base_url = 'https://github.com/ieee8023/latentshift/releases/download/weights/'
weights_path = utils.get_cache_folder()

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
    """A classifier trained on celeba attributes

    Branched Multi-Task Networks: Deciding What Layers To Share
    Simon Vandenhende, Stamatios Georgoulis, Bert De Brabandere, Luc Van Gool
    British Machine Vision Virtual Conference
    https://arxiv.org/abs/1904.02920
    """
    def __init__(self, download=False):
        super().__init__()

        filename = weights_path + "BranchedTiny.ckpt"
        url = "https://github.com/ieee8023/latentshift/releases/download/weights/BranchedTiny.ckpt"
        
        if not os.path.isfile(filename):
            if download:
                utils.download(url, filename)
            else:
                print("No weights found, specify download=True to download them.")
        
        self.model = BranchedTiny.BranchedTiny(filename)
        self.model = self.model.eval()
        
        self.targets = self.model.attributes
        self.attributes = self.model.attributes
        
    def forward(self, x):
        return self.model(x)


def _bird_apply_test_transforms(inp):
    out = torchvision.transforms.functional.resize(inp, [224,224])
    out = torchvision.transforms.functional.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return out
    
class BirdClassifier(torch.nn.Module):
    """
    From: https://www.kaggle.com/code/sharansmenon/pytorch-cubbirds200-classification/notebook
    Test Accuracy (Overall): 87%
    """
    def __init__(self, download=False):
        super().__init__()

        filename = weights_path + "cub2011-deit-tiny.zip"
        url = "https://github.com/ieee8023/latentshift/releases/download/weights/cub2011-deit-tiny.zip"
        
        if not os.path.isfile(filename):
            if download:
                utils.download(url, filename)
            else:
                print("No weights found, specify download=True to download them.")
        

        self.model = torch.jit.load(filename)
        self.model = self.model.eval()

        self.targets = [
            'Black_footed_Albatross',
            'Laysan_Albatross',
            'Sooty_Albatross',
            'Groove_billed_Ani',
            'Crested_Auklet',
            'Least_Auklet',
            'Parakeet_Auklet',
            'Rhinoceros_Auklet',
            'Brewer_Blackbird',
            'Red_winged_Blackbird',
            'Rusty_Blackbird',
            'Yellow_headed_Blackbird',
            'Bobolink',
            'Indigo_Bunting',
            'Lazuli_Bunting',
            'Painted_Bunting',
            'Cardinal',
            'Spotted_Catbird',
            'Gray_Catbird',
            'Yellow_breasted_Chat',
            'Eastern_Towhee',
            'Chuck_will_Widow',
            'Brandt_Cormorant',
            'Red_faced_Cormorant',
            'Pelagic_Cormorant',
            'Bronzed_Cowbird',
            'Shiny_Cowbird',
            'Brown_Creeper',
            'American_Crow',
            'Fish_Crow',
            'Black_billed_Cuckoo',
            'Mangrove_Cuckoo',
            'Yellow_billed_Cuckoo',
            'Gray_crowned_Rosy_Finch',
            'Purple_Finch',
            'Northern_Flicker',
            'Acadian_Flycatcher',
            'Great_Crested_Flycatcher',
            'Least_Flycatcher',
            'Olive_sided_Flycatcher',
            'Scissor_tailed_Flycatcher',
            'Vermilion_Flycatcher',
            'Yellow_bellied_Flycatcher',
            'Frigatebird',
            'Northern_Fulmar',
            'Gadwall',
            'American_Goldfinch',
            'European_Goldfinch',
            'Boat_tailed_Grackle',
            'Eared_Grebe',
            'Horned_Grebe',
            'Pied_billed_Grebe',
            'Western_Grebe',
            'Blue_Grosbeak',
            'Evening_Grosbeak',
            'Pine_Grosbeak',
            'Rose_breasted_Grosbeak',
            'Pigeon_Guillemot',
            'California_Gull',
            'Glaucous_winged_Gull',
            'Heermann_Gull',
            'Herring_Gull',
            'Ivory_Gull',
            'Ring_billed_Gull',
            'Slaty_backed_Gull',
            'Western_Gull',
            'Anna_Hummingbird',
            'Ruby_throated_Hummingbird',
            'Rufous_Hummingbird',
            'Green_Violetear',
            'Long_tailed_Jaeger',
            'Pomarine_Jaeger',
            'Blue_Jay',
            'Florida_Jay',
            'Green_Jay',
            'Dark_eyed_Junco',
            'Tropical_Kingbird',
            'Gray_Kingbird',
            'Belted_Kingfisher',
            'Green_Kingfisher',
            'Pied_Kingfisher',
            'Ringed_Kingfisher',
            'White_breasted_Kingfisher',
            'Red_legged_Kittiwake',
            'Horned_Lark',
            'Pacific_Loon',
            'Mallard',
            'Western_Meadowlark',
            'Hooded_Merganser',
            'Red_breasted_Merganser',
            'Mockingbird',
            'Nighthawk',
            'Clark_Nutcracker',
            'White_breasted_Nuthatch',
            'Baltimore_Oriole',
            'Hooded_Oriole',
            'Orchard_Oriole',
            'Scott_Oriole',
            'Ovenbird',
            'Brown_Pelican',
            'White_Pelican',
            'Western_Wood_Pewee',
            'Sayornis',
            'American_Pipit',
            'Whip_poor_Will',
            'Horned_Puffin',
            'Common_Raven',
            'White_necked_Raven',
            'American_Redstart',
            'Geococcyx',
            'Loggerhead_Shrike',
            'Great_Grey_Shrike',
            'Baird_Sparrow',
            'Black_throated_Sparrow',
            'Brewer_Sparrow',
            'Chipping_Sparrow',
            'Clay_colored_Sparrow',
            'House_Sparrow',
            'Field_Sparrow',
            'Fox_Sparrow',
            'Grasshopper_Sparrow',
            'Harris_Sparrow',
            'Henslow_Sparrow',
            'Le_Conte_Sparrow',
            'Lincoln_Sparrow',
            'Nelson_Sharp_tailed_Sparrow',
            'Savannah_Sparrow',
            'Seaside_Sparrow',
            'Song_Sparrow',
            'Tree_Sparrow',
            'Vesper_Sparrow',
            'White_crowned_Sparrow',
            'White_throated_Sparrow',
            'Cape_Glossy_Starling',
            'Bank_Swallow',
            'Barn_Swallow',
            'Cliff_Swallow',
            'Tree_Swallow',
            'Scarlet_Tanager',
            'Summer_Tanager',
            'Artic_Tern',
            'Black_Tern',
            'Caspian_Tern',
            'Common_Tern',
            'Elegant_Tern',
            'Forsters_Tern',
            'Least_Tern',
            'Green_tailed_Towhee',
            'Brown_Thrasher',
            'Sage_Thrasher',
            'Black_capped_Vireo',
            'Blue_headed_Vireo',
            'Philadelphia_Vireo',
            'Red_eyed_Vireo',
            'Warbling_Vireo',
            'White_eyed_Vireo',
            'Yellow_throated_Vireo',
            'Bay_breasted_Warbler',
            'Black_and_white_Warbler',
            'Black_throated_Blue_Warbler',
            'Blue_winged_Warbler',
            'Canada_Warbler',
            'Cape_May_Warbler',
            'Cerulean_Warbler',
            'Chestnut_sided_Warbler',
            'Golden_winged_Warbler',
            'Hooded_Warbler',
            'Kentucky_Warbler',
            'Magnolia_Warbler',
            'Mourning_Warbler',
            'Myrtle_Warbler',
            'Nashville_Warbler',
            'Orange_crowned_Warbler',
            'Palm_Warbler',
            'Pine_Warbler',
            'Prairie_Warbler',
            'Prothonotary_Warbler',
            'Swainson_Warbler',
            'Tennessee_Warbler',
            'Wilson_Warbler',
            'Worm_eating_Warbler',
            'Yellow_Warbler',
            'Northern_Waterthrush',
            'Louisiana_Waterthrush',
            'Bohemian_Waxwing',
            'Cedar_Waxwing',
            'American_Three_toed_Woodpecker',
            'Pileated_Woodpecker',
            'Red_bellied_Woodpecker',
            'Red_cockaded_Woodpecker',
            'Red_headed_Woodpecker',
            'Downy_Woodpecker',
            'Bewick_Wren',
            'Cactus_Wren',
            'Carolina_Wren',
            'House_Wren',
            'Marsh_Wren',
            'Rock_Wren',
            'Winter_Wren',
            'Common_Yellowthroat',
            ]
        
    def forward(self, x):
        x = _bird_apply_test_transforms(x)
        return self.model(x)

        
        
class ImageNetClassifier(torch.nn.Module):
    def __init__(self, download=False):
        super().__init__()


        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=self.weights).eval()

        self.targets = self.weights.meta['categories']
        self.targets = [t.replace(' ', '_') for t in self.targets]
        
    def forward(self, x):
        x = self.weights.transforms()(x)
        return self.model(x)


class WaterbirdClassifier(torch.nn.Module):
    def __init__(self, weights = 'baseline', download=False):
        """weights = 'baseline', 'place', 'withdro'
        """
        super().__init__()

        available_weights = ['baseline', 'place', 'withdro']

        if weights.startswith('/'):
            # if full path specified
            weights_ckpt = weights
        else:
            if not weights in available_weights:
                raise Exception(f'weights must be one of {available_weights}')
                
            weights = 'waterbirds_' + weights + '.pth'
            if (not os.path.isfile(weights_path + weights)):
                if download:
                    utils.download(base_url + weights, weights_path + weights)
                else:
                    print("No weights found, specify download=True to download them.")
            
            weights_ckpt = weights_path + weights
        
        self.model = torch.load(weights_ckpt, map_location='cpu')
        self.targets = ['Landbird', 'Waterbird']

        scale = 256.0/224.0
        target_resolution = 224
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((int(target_resolution*scale), int(target_resolution*scale))),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def forward(self, x):
        x = self.transform(x)
        return self.model(x)


    












    



        