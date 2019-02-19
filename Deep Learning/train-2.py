import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import vgg19, densenet121, vgg16
from torchvision import datasets, models, transforms
import torchvision
from torch import nn, optim
import torch
import torch.nn.functional as F
from collections import OrderedDict
from helper import validation 

parser = argparse.ArgumentParser(description='Training a neural network to recognize flowers')
parser.add_argument('data_directory', type=str, help='Directory containing data' , default='data_dir')
parser.add_argument('--gpu', type=bool, default=True, help='Using GPU or CPU')
parser.add_argument('--arch', type=str, default='vgg19', help='VGG or Densenet')
parser.add_argument('--epochs', type=str, default=5, help='How many epochs to use during training')
parser.add_argument('--lr', type=str, default=0.0001, help='What learning rate to use for training')
parser.add_argument('--hidden_sizes', type=list, default=[2048,1024], help='Hidden layer sizes')
args = parser.parse_args()

device = 'cuda' if args.gpu == True else 'cpu'

train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'
test_dir = args.data_directory + '/test'

# : Define your transforms for the training, validation, and testing sets
data_transforms = {'train' : transforms.Compose([transforms.RandomRotation(30),
                                                 transforms.RandomResizedCrop(224),                                                 
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225]),
                                                 ]),
                   
                   'valid' : transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])
                                                ]),
                   
                   'test' : transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225]),
                                               ])
                  }
                                                
# : Load the datasets with ImageFolder
image_datasets = {'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                  'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                  'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test'])
                 }

# : Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train' : DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
               'valid' : DataLoader(image_datasets['valid'], batch_size=32),
               'test' : DataLoader(image_datasets['test'], batch_size=32)
              }

input_size = 25088 if args.arch == 'vgg19' else 1024
hidden_sizes = [2048, 1024]
output_size = 102

hidden_layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])

layers_map = [('fc1', nn.Linear(input_size, hidden_sizes[0])),
                    ('relu1', nn.ReLU()),
                    ('dropout1', nn.Dropout(0.2))]
count = 1
for h1, h2 in hidden_layer_sizes:
    count +=1
    layers_map.append(('fc'+str(count), nn.Linear(h1,h2)))
    layers_map.append(('relu'+str(count), nn.ReLU()))
    layers_map.append(('dropout'+str(count), nn.Dropout(0.2)))
layers_map.append(('fc'+str(count+1), nn.Linear(hidden_sizes[-1], output_size)))
layers_map.append(('output', nn.LogSoftmax(dim=1)))
classifier = nn.Sequential(OrderedDict(layers_map))

model = getattr(models, args.arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = classifier
if args.gpu:
    model.cuda()
else:
    model.cpu()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)

model.train()

print_every = 40
steps = 0

for epoch in range(args.epochs):
    accuracy = 0
    running_loss = 0.0
    for inputs, labels in dataloaders['train']:

        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(inputs)
        _, preds = torch.max(output.data, 1)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        if steps % print_every == 0:
            test_loss, test_accuracy = validation(model, dataloaders['valid'], criterion, device)
            print("Epoch: {}/{}".format(epoch+1, args.epochs),
                  "Train Loss: {:.4f}".format(running_loss/print_every),
                  "Train Accuracy : {:.4f}".format(accuracy/print_every),
                  "Validation Loss : {:.4f}".format(test_loss),
                  "Validation Accuracy : {:.4f}".format(test_accuracy))
            model.train()
            accuracy = 0
            running_loss = 0

# Do validation on the test set, print results
test_loss, test_accuracy = validation(model, dataloaders['test'], criterion, device)
print("Test Loss : {:.4f}".format(test_loss),
    "Test Accuracy : {:.4f}".format(test_accuracy))

# Save the checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'arch' : args.arch,
              'classifier' : model.classifier,
              'state_dict' : model.state_dict(),
              'optimizer' : optimizer,
              'optimizer_dict' : optimizer.state_dict(),
              'epochs' : args.epochs,
              'class_to_idx' : model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')
