import json
import pandas as pd
import csv

def counts_to_ids(counts_csv:str,out_json:str):
    '''Reads the count csv and uses it to convert the species names into integers for mapping'''
    species = []
    with open(counts_csv,'r',encoding='utf-8') as f:
        r =csv.DictReader(f)
        for entry in r:
            species.append(entry['label'])
    
    species_to_id = {sp: i for i,sp in enumerate(species)}
    
    with open(out_json,'w',encoding='utf-8') as f:
        json.dump(species_to_id,f,ensure_ascii=False,indent=2)   
        
    return species_to_id
 