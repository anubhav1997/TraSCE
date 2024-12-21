import torch
import random
import pandas as pd
import argparse
import os
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from transformers import CLIPTokenizer
from functools import reduce
import operator
import time
import tqdm
import json
import numpy as np
import pickle
from torchvision.models import vit_h_14, ViT_H_14_Weights, resnet50, ResNet50_Weights
from PIL import Image
import argparse 

def parse_args():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="",
    )
    
    parser.add_argument(
        "--target_class",
        type=str,
        default="",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="temp_save",
    )
    return parser.parse_args()
    

args = parse_args()



def image_classify(images_path, prompts_path, save_path, target_class, device='cuda', topk=1, batch_size=200):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.to(device)
    model.eval()

    scores = {}
    categories = {}
    indexes = {}
    for k in range(1,topk+1):
        scores[f'top{k}']= []
        indexes[f'top{k}']=[]
        categories[f'top{k}']=[]

    names = os.listdir(images_path)
    names = [name for name in names if '.png' in name or '.jpg' in name]
    if len(names) == 0:
        images_path = images_path+'/imgs'
        names = os.listdir(images_path)
        names = [name for name in names if '.png' in name or '.jpg' in name]

    preprocess = weights.transforms()

    images = []
    for name in names:
        img = Image.open(os.path.join(images_path,name))
        batch = preprocess(img)
        images.append(batch)

    if batch_size == None:
        batch_size = len(names)
    if batch_size > len(names):
        batch_size = len(names)
    images = torch.stack(images)
    # Step 4: Use the model and print the predicted category
    for i in range(((len(names)-1)//batch_size)+1):
        batch = images[i*batch_size: min(len(names), (i+1)*batch_size)].to(device)
        with torch.no_grad():
            prediction = model(batch).softmax(1)
        probs, class_ids = torch.topk(prediction, topk, dim = 1)

        for k in range(1,topk+1):
            scores[f'top{k}'].extend(probs[:,k-1].detach().cpu().numpy())
            indexes[f'top{k}'].extend(class_ids[:,k-1].detach().cpu().numpy())
            categories[f'top{k}'].extend([weights.meta["categories"][idx] for idx in class_ids[:,k-1].detach().cpu().numpy()])

    if save_path is not None:
        df = pd.read_csv(prompts_path)
        df['case_number'] = df['case_number'].astype('int')
        case_numbers = []
        for i, name in enumerate(names):
            case_number = name.split('/')[-1].split('_')[0].replace('.png','').replace('.jpg','')
            case_numbers.append(int(case_number) + 1 ) #adding a +1 here. Should be removed later 

        dict_final = {'case_number': case_numbers}

        for k in range(1,topk+1):
            dict_final[f'category_top{k}'] = categories[f'top{k}'] 
            dict_final[f'index_top{k}'] = indexes[f'top{k}'] 
            dict_final[f'scores_top{k}'] = scores[f'top{k}'] 


        print("hereee:::::", len(categories[f'top1']), len(case_numbers))
        df_results = pd.DataFrame(dict_final)
        merged_df = pd.merge(df,df_results)
        merged_df.to_csv(save_path)

        # compute the accuracy of the target class and others
        target_acc = 0
        other_acc = 0
        for i in range(len(merged_df)):
            if merged_df['category_top1'][i].lower() == merged_df['class'][i] and merged_df['class'][i] == target_class:
                target_acc += 1
            elif merged_df['category_top1'][i].lower() == merged_df['class'][i] and merged_df['class'][i] != target_class:
                other_acc += 1
        
        target_acc /= (len(merged_df)/10.)  # imagenette has 10 classes
        other_acc /= (9*len(merged_df)/10.)
        
        print(target_acc, other_acc)
        
        return target_acc, other_acc


print(image_classify(args.path, args.prompts, args.save_path, args.target_class))

