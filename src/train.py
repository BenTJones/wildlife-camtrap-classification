import torch
from src.data.dataset import CCTImageDataset
import pandas as pd
from torch.nn import CrossEntropyLoss
import csv

def setup_loss_log(out:str):
    with open(out,'w',newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        

def train_one_epoch(model,dataloader,optimiser,loss_fn,device):
    train_loss = 0.0
    total_correct = 0
    length = len(dataloader.dataset)
    
    model.train()
    for imgs,lbls in dataloader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        
        optimiser.zero_grad()
        predictions = model(imgs)
        error = loss_fn(predictions,lbls)
        error.backward()
        optimiser.step()
        _,class_pred = torch.max(predictions,1) #Output is 2d vector of value and index. Kwargs are tensor and dimension for inspection
        total_correct += (class_pred == lbls).sum().item()
        train_loss += error.item()
    
    train_acc = total_correct / length
    train_loss = train_loss / length
    return train_loss, train_acc

@torch.no_grad()    
def val(model,dataloader,loss_fn,device):
    model.eval()
    val_loss = 0.0
    total_correct = 0
    length = len(dataloader.dataset)
    
    for imgs,lbls in dataloader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        
        preds = model(imgs)
        class_pred = torch.max(preds,1)
        
        loss = loss_fn(preds,lbls)
        val_loss += loss.item()
        _,class_pred = torch.max(preds,1)
        total_correct += (class_pred == lbls).sum().item()
    avg_val_loss = val_loss / length
    val_acc = total_correct/ length
    return avg_val_loss,val_acc


def run(train_loader,val_loader,model,lr,epochs,patience,log_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = CrossEntropyLoss()
    optimiser = torch.optim.Adam(params = model.parameters(),lr= lr)
    best_val_loss = float('inf')
    
    for e in range(epochs):
        train_loss,train_acc = train_one_epoch(model,train_loader,optimiser,loss_fn,device)
        val_loss,val_acc = val(model,val_loader,loss_fn,device)
        print(f'Train Loss: {train_loss:.5f} & Train Accuracy: {train_acc*100:.3f}')
        print(f'Val Loss: {val_loss:.5f} & Val Accuracy: {val_acc*100:.3f}')
        
        with open(log_path, mode="a", newline="",encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([e, train_loss, train_acc, val_loss, val_acc])
        
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),'best_model.pth')
            patience_counter = 0
        
        else:
            patience_counter += 1
            if patience_counter == patience:
                break