import csv,os,pandas
from torch.utils.data import Dataset
from torchvision import transforms as t

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
#Represent what transformations need to be done to normalise the images into a suitable format for the pytroch model

def transform_images(train:bool,size:int = 224):
    if train:
        return t.Compose([
            t.Resize(size,size),
            t.RandomHorizontalFlip(p = 0.5),
            t.ToTensor(), #introduce flip onto train to make sure its not rotation specific patterns
            t.Normalize(IMAGENET_MEAN,IMAGENET_STD)
            
        ])
    else:
        return t.Compose([
            t.Resize(size,size),
            t.ToTensor(),
            t.Normalize(IMAGENET_MEAN,IMAGENET_STD)
        ])

class CCTImageDataset(Dataset): #Tells Dataloader how to access and transform images before passing into model
    def __init__(self,maifest_df,image_dir,labels,transform):
       self.image_dir = image_dir
       self.transform = transform 
       
       