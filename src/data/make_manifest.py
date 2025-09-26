import os,json,csv, argparse
from collections import Counter

def load_coco(path : str)-> dict: #Term for image datasets, common object in context
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)
    
def make_manifest_rows(img_path:str,coco:dict):
    img_by_id = {img['id'] :  img for img in coco['images']}
    cat_by_id = {cat['id'] :cat['name'] for cat in coco['categories']}
    
    label_by_image = {}
    
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in label_by_image:
            label_by_image[img_id] = cat_by_id[ann['category_id']]
        
    rows,missing = [],[]
    
    for img_id, img in img_by_id.items():
        rel = img['file_name'].replace('/',os.sep) #Ensure it works on diff os 
        path = os.path.join(img_path, rel)
        if os.path.exists(path):
            label = label_by_image.get(img_id,'empty')
            location = img.get('location')
            rows.append((rel,label,location))
        else:
            missing.append(rel)
            
    return rows,missing #Returns missing file path, and rows : list of tuples containing rel path, location(IRL) and the label

def write_manifest(rows,csv_name:str):
    os.makedirs(os.path.dirname(csv_name),exist_ok= True)
    with open(csv_name,'w',newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rel_path','label','location'])
        w.writerows(rows)
        
def write_counts(rows,csv_name:str):
    counts = Counter(r[1] for r in rows) #Counts all instances of each label given by row index 1
    with open(csv_name,'w',newline='',encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['label','count'])
        for k,v in sorted(counts.items(),key = lambda x : (-x[1],x[0])): #Ensures ordering via descending count from key val tuple
            w.writerow([k,v])
            
def main(images,annotations,manifest_path,counts_path,missing_path):
    coco = load_coco(annotations)
    rows,missing = make_manifest_rows(images,coco)
    write_manifest(rows,manifest_path)
    write_counts(rows,counts_path)
    if missing:
        with open(missing_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing))
    print(f'Manifest: {manifest_path}, Rows:{len(rows)}')
    print(f'Class Counts : {counts_path}')
    if missing:
        print(f'Missing Files: {missing_path}')
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-images", default="data/cct/images")
    ap.add_argument("-annotations", default="data/cct/annotations/caltech_images_20210113.json")
    ap.add_argument("-manifest-path", default="data/cct/manifest.csv")
    ap.add_argument("-counts-path",default="data/cct/class_counts.csv")
    ap.add_argument("-missing-path", default="data/cct/missing_files.txt")
    args = ap.parse_args()
    main(args.images, args.annotations, args.manifest_path, args.counts_path, args.missing_path)