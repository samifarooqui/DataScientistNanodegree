import json
import torch
from PIL import Image
from helper import process_image, load_checkpoint
from collections import OrderedDict
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Predicting the flower')
parser.add_argument('--checkpoint', type=str, help='Path to checkpoint' , default='checkpoint.pth')
parser.add_argument('--image_path', type=str, help='Path to file' , default='flowers/test/28/image_05230.jpg')
parser.add_argument('--gpu', type=bool, default=True, help='GPU or CPU')
parser.add_argument('--topk', type=int, help='K prediction' , default=0)
args = parser.parse_args()

image_path = args.image_path
with open(args.cat_to_name_json, 'r') as f:
    cat_to_name = json.load(f)

model, checkpoint = load_checkpoint(args.checkpoint)

im = Image.open(image_path)
processed_image = process_image(im)

def predict(image_path, model, topk=5, device='cuda'):
    im = Image.open(image_path)
    processed_image = process_image(im).unsqueeze(0)
    model.to(device)
    model.eval()    
    with torch.no_grad():
        processed_image = processed_image.to(device).float()
        output = model(processed_image)
        ps = torch.exp(output)
    pred = ps.topk(topk)
    flower_ids = pred[1][0].to('cpu')
    flower_ids = torch.Tensor.numpy(flower_ids)
    probs = pred[0][0].to('cpu')
    idx_to_class = {k:v for v,k in checkpoint['class_to_idx'].items()}
    flower_names = np.array([cat_to_name[idx_to_class[x]] for x in flower_ids])
        
    return probs, flower_names

if args.topk:
    probs, flower_names = predict(image_path, model, args.topk,'cuda' if args.gpu else 'cpu')
    print('Most probable {} flowers:'.format(args.topk))
    for i in range(args.topk):
        print('{} : {:.2f}'.format(flower_names[i],probs[i]))
else:
    probs, flower_names = predict(image_path, model)
    print('Flower prediction {} with {:.2f} probability'.format(flower_names[0], probs[0]))
    