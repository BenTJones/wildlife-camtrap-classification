from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
import torch

def compute_preds(test_loader,model,device):
    model.eval()
    y_true,y_pred = [],[] 
    
    for imgs,lbls in test_loader:
        imgs,lbls = imgs.to(device),lbls.to(device)
        preds = model(imgs)
        _,class_preds = torch.max(preds,dim=1)
        y_true.extend(lbls)
        y_pred.extend(class_preds)
        
    return y_true,y_pred