import cv2
import io
from PIL import Image

import os
import pickle
import json

import pandas as pd



def img_to_bytes(img_path):
    # read img
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # Convert the OpenCV image (numpy array) to a PIL Image
    pil_img = Image.fromarray(img)
    # Create a BytesIO object
    img_stream = io.BytesIO()
    # Save the image to the BytesIO object in jpg
    pil_img.save(img_stream, format='PNG')
    # Retrieve the byte data
    img_bytes = img_stream.getvalue()
    return img_bytes

data = {"jpg": [], "prompt": []}

print('reading to df...')
for files in os.walk('../data/'):
    
    for f in files[2]:
        
        if f.endswith('.jpg'):
            f = os.path.join('../data/', f)
            print(f'   reading {f}')
            data["jpg"].append(img_to_bytes(f))
        if f.endswith('.json'):
            f = os.path.join('../data/', f)
            print(f'   reading {f}')
            with open(f, 'r') as json_file:
                prompt_data = json.load(json_file)
            data["prompt"].append(prompt_data['prompt'])


print('saving parquet...')
df = pd.DataFrame(data)
df.to_parquet('data.parquet')

print('checking length...')
print(f'   len(df)={len(df)}')
p = pd.read_parquet('data.parquet')
print(f'   len(p)={len(p)}')

print('down')