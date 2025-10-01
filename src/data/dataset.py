import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as t
from PIL import Image



#Represent what transformations need to be done to normalise the images into a suitable format for the pytroch model

def transform_images(train:bool,size:int = 224):
    '''Makes the image transform for train and test + val.
    Returns a transformation pipeline for each image depengin on Split'''
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    if train:
        return t.Compose([
            t.Resize((size,size)),
            t.RandomHorizontalFlip(p = 0.5),
            t.ToTensor(), #introduce flip onto train to make sure its not rotation specific patterns
            t.Normalize(IMAGENET_MEAN,IMAGENET_STD)
            
        ])
    else:
        return t.Compose([
            t.Resize((size,size)),
            t.ToTensor(),
            t.Normalize(IMAGENET_MEAN,IMAGENET_STD)
        ])

class CCTImageDataset(Dataset): #Tells Dataloader how to access and transform images before passing into model
    def __init__(self,df,image_dir,class_to_idx,transform = False):
       self.image_dir = image_dir
       self.transform = transform_images(transform)
       self.class_to_idx = class_to_idx
       
       df['full_path'] = self.image_dir + os.sep + df['rel_path']
       df['int_label'] = df['label'].map(self.class_to_idx)
       
       self.image_data = list(zip(df['full_path'],df['int_label'],df['location']))
       
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, index):
        img_path ,label,_ = self.image_data[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image,label
    
def create_dataloader(dataset, batch_size, shuffle:bool, num_workers=0):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )