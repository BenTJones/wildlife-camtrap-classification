import timm
import json

def create_model(labels_path:str):
    with open(labels_path,'r',encoding= 'utf-8') as l:
        labels = json.load(l)        
    num_classes = len(labels)    
    
    model = timm.create_model(
        model_name = 'efficientnet_b0',
        pretrained= True,
        num_classes = num_classes
    )
    return model

