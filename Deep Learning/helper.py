import numpy as np
import torchvision
from torchvision import datasets, models
import torch
import argparse

def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    for param in model.parameters():
        param.requires_grad = False
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    return model, checkpoint

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    model.to(device)
    model.eval()
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        output = model(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss/len(testloader), accuracy/len(testloader)

def process_image(image):
    image = image.resize((round(256*image.size[0]/image.size[1]) if image.size[0]>image.size[1] else 256,
                          round(256*image.size[1]/image.size[0]) if image.size[1]>image.size[0] else 256))  
    
    image = image.crop((image.size[0]/2-224/2, image.size[1]/2-224/2, image.size[0]/2+224/2, image.size[1]/2+224/2))

    np_image = (np.array(image)/255-[0.485,0.456,0.406])/[0.229, 0.224, 0.225]
    np_image = np_image.transpose((2,0,1))
    
    return torch.from_numpy(np_image)
