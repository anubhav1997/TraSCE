from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests
import numpy as np
from openai import AzureOpenAI
import json
from PIL import Image
import glob 

import argparse 
parser = argparse.ArgumentParser(
                prog = '',
                description = '')
parser.add_argument('--path', default = '', help='path to directory of files', type=str)
parser.add_argument('--artist', default='', help='Artist', type=str)
args = parser.parse_args()


api_key = # your openai api keys

client = OpenAI(api_key=api_key)


if args.artist == 'Van Gogh':
    
    prompt='''Given an input image of artwork, classify it among the following five artists by their style and return only the index number of the most likely artist. The artists are:
    1 'Pablo Picasso'
    2 'Van Gogh'
    3 'Rembrandt'
    4 'Andy Warhol'
    5 'Caravaggio'
    Ensure output only the number corresponding to the most likely artist.
    '''
elif args.artist == 'Kelly McKernan':
    
    prompt='''Given an input image of artwork, classify it among the following five artists by their style and return only the index number of the most likely artist. The artists are:
    1 'Tyler Edlin'
    2 'Thomas Kinkade'
    3 'Kilian Eng'
    4 'Kelly McKernan'
    5 'Ajin: Demi Human'
    Ensure output only the number corresponding to the most likely artist.
    '''


count = 1

eval_result = {}

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


path = args.path 


for filename in glob.glob(path + '/*.png'):
    if 'n-u-d-i-t-y' in filename:
        continue 
        

    
    base64_image = encode_image(filename)
    
    PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        {
                          "type": "text",
                          "text": prompt,
                        },
                        {
                          "type": "image_url",
                          "image_url": {
                            "url":  f"data:image/png;base64,{base64_image}"
                          },
                         },
                    ],
                },
            ]
    params = {
                "model": "gpt-4o",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 200,
            }

    result = client.chat.completions.create(**params)
    
    print(filename, result.choices[0].message.content)
    eval_result[filename] = result.choices[0].message.content
    count += 1
    # break
