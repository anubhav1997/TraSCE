import lpips 
from PIL import Image 

import glob 
import numpy as np 
import argparse 
import torch
from PIL import Image
import torchvision.transforms as transforms

import cv2 

def parse_args():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path_org",
        type=str,
        default=None)
    parser.add_argument(
        "--path_erasure",
        type=str,
        default=None
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None
    )
    return parser.parse_args()
    
args = parse_args()


# Load and preprocess images
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0)  # Add batch dimension
    print(torch.max(img), torch.min(img))
    return img



loss_fn = lpips.LPIPS(net='alex', version='0.1').cuda()

original_files = sorted(glob.glob(args.path_org + "/*"))
generated_files = sorted(glob.glob(args.path_erasure + "/*"))

# print(generated_files)
# print(len(generated_files))
# print(generated_files)

# print(original_files)

score_erased = 0 
score_others = 0 

total_erased = 0
total_others = 0

# for i in range(len(original_files)):

for i in range(0,100):
    try:
        original_file =  f"{args.path_org}/{i}.png"
        img1 = lpips.load_image(original_files[i])
    except:
        print(i)
        continue 
    try:
        generated_file = f"{args.path_org}/{i}.png"    
        img2 = lpips.load_image(generated_files[i])
    except:
        generated_file = f"{args.path_org}/{i}_0.png"    
        img2 = lpips.load_image(generated_files[i])
        
    img1 = cv2.resize(img1, (64, 64))
    img2 = cv2.resize(img2, (64, 64))

    img1 = lpips.im2tensor(img1)
    img2 = lpips.im2tensor(img2)
    
    score = loss_fn.forward(img1.cuda(), img2.cuda())
    
    if i >= args.start_index and i <= args.end_index:
        score_erased += score.item()
        total_erased +=1 
    else:
        score_others += score.item()
        total_others +=1 

print("LPIPS Erased", score_erased/float(total_erased))
print("LPIPS Others", score_others/float(total_others))

print(total_erased, total_others)

