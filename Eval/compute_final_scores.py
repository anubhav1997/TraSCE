import numpy as np 
import pandas as pd 



import argparse 
parser = argparse.ArgumentParser(
                prog = '',
                description = '')
parser.add_argument('--path', default = '', help='path to directory of files', type=str)
parser.add_argument('--artist', default='', help='Artist', type=str)
args = parser.parse_args()


universal = 0
total_universal = 0

erased = 0
total_erased = 0

txtfile = args.path 

if args.artist == 'Kelly McKernan':
    index = 4
elif args.artist == 'Van Gogh':
    index = 2
    
    

with open(txtfile, 'r') as infile:
    stripped = (line.strip() for line in infile)
    lines = (line.split(" ") for line in stripped if line)
    
    
    for line in lines:
        try:
            label = int(line[-1]) 
        except:
            print(line[-1])
            continue
        # continue
    
        print(line[-2].split('/')[-1].split('.png')[0].split('_')[0])
        if int(line[-2].split('/')[-1].split('.png')[0].split('_')[0]) <=19:
            y = 1
        elif int(line[-2].split('/')[-1].split('.png')[0].split('_')[0]) >=20 and int(line[-2].split('/')[-1].split('.png')[0].split('_')[0]) <=39:
            y = 2
        elif int(line[-2].split('/')[-1].split('.png')[0].split('_')[0]) >=40 and int(line[-2].split('/')[-1].split('.png')[0].split('_')[0]) <=59:
            y = 3
        elif int(line[-2].split('/')[-1].split('.png')[0].split('_')[0]) >=60 and int(line[-2].split('/')[-1].split('.png')[0].split('_')[0]) <=79:
            y = 4
        elif int(line[-2].split('/')[-1].split('.png')[0].split('_')[0]) >=80 and int(line[-2].split('/')[-1].split('.png')[0].split('_')[0]) <=100:
            y = 5

        
        if y !=index:
            total_universal +=1
            universal += int(y==label)
        elif y==index:
            total_erased +=1
            erased += int(y==label)

print(universal/float(total_universal))
print(erased/float(total_erased))

print(erased,total_erased,universal,total_universal)

