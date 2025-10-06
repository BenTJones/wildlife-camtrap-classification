import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import os

def load_manifest(manifest:str) -> pd.DataFrame:
    return pd.read_csv(manifest)
    
def split_df(df:pd.DataFrame,train_frac=0.8):
    '''Performs Stratified Group K fold splitting onto a df with the grouping of location and labels
    Exprects input of train_frac and returns Dataframes of size approximately Train fraction for train and half the remainder for val,test'''
    seed = 10 #Sets constant seed to ensure consistent randoms
    y = df['label']
    gr = df['location']
    
    skf = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state = seed)
    train_length = int(len(df)*train_frac)
    candidates = list(skf.split(df,y,gr))
    train_idx,temp_idx = min(candidates, key = lambda cand: abs(len(cand[0])- train_length))#GIves closest split to 80/20
    train = df.iloc[train_idx]
    temp = df.iloc[temp_idx]
    
    
    skf2 = StratifiedGroupKFold(n_splits=2,shuffle=True,random_state = seed)
    y2 = temp['label']
    gr2 = temp['location']
    val_idx, test_idx = next(skf2.split(temp,y2,gr2))
    
    val = temp.iloc[val_idx]
    test = temp.iloc[test_idx]
    return train,val,test

def summarise_split(split: pd.DataFrame,name:str):
    n = len(split)
    empty = (split['label'] == 'empty').sum()
    stats = {
        "split": name,
        "rows": n,
        "cols": split.shape[1],
        "n_classes": split['label'].nunique(),
        "n_groups": split['location'].nunique(),
        "empty_count": empty,
        "empty_pct": round(100 * empty / n, 2)
    }
    return stats

def save_splits(train,val,test, out_dir = 'data/cct/splits'):
    os.makedirs(out_dir,exist_ok= True)
    train.to_csv(os.path.join(out_dir,'train.csv'),index= False)
    test.to_csv(os.path.join(out_dir,'test.csv'),index= False)
    val.to_csv(os.path.join(out_dir,'val.csv'),index= False)
    
from sklearn.model_selection import train_test_split

def make_subset(train:pd.DataFrame,frac : int):
    '''Splits dataset allowing for running on local machine'''
    sub,_ = train_test_split(
        train,
        train_size= frac,
        shuffle= True,
        stratify= train['label']
    )
    return sub
