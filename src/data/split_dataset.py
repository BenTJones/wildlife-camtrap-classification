import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def load_manifest(manifest:str) -> pd.DataFrame:
    with open(manifest,'r',encoding= 'utf-8') as f:
        df = pd.read_csv(f)
    return df
    
def split_df(df:pd.DataFrame,train_frac=0.8):
    '''Performs Stratified Group K fold splitting onto a df with the grouping of location and labels
    Exprects input of train_frac and returns Dataframes of size approximately Train fraction for train and half the remainder for val,test'''
    seed = 42 #Sets constant seed to ensure consistent randoms
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

