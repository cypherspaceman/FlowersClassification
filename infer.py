import argparse
import json
import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn.functional as F

from datetime import datetime
from PIL import Image
from torch import nn
from torchvision import datasets, transforms, models

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the transforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)


class Classifier(nn.Module):
    """Classifier model"""
    def __init__(self, hu):
        super().__init__()
        self.fc1 = nn.Linear(25088, hu)
        self.fc2 = nn.Linear(hu, 512)
        self.fc3 = nn.Linear(512, 102)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # no dropout for output
        x = F.log_softmax(self.fc3(x), dim=1)

        return x


def create_model(model_name, lr=0.003, hu=4096):
    """Create an instance of the model"""
    if model_name == 'VGG':
        model = models.vgg16(pretrained=True)    
    elif model_name == 'Densenet':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f'Unknown model name {model_name}') 


    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = Classifier(hu)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr)

    return model, optimizer


def train_network(checkpoint_base='FlowersClassification/checkpoints/checkpoint', model_name='vgg11', lr=0.003, hu=4096, epochs=20, device="cpu", continue_from_checkpoint=None):
    """Train the network, and provide feedback on progress"""

    if continue_from_checkpoint is not None:
        model, model_name, optimizer, epoch, train_loss, validation_loss, validation_accuracy = load_checkpoint(continue_from_checkpoint, lr, hu)
    else:
        model, optimizer = create_model(model_name, lr, hu)
        epoch = 0
        train_loss = []
        validation_loss = []
        validation_accuracy = []

    model.to(device)

    # Set loss function
    criterion = torch.nn.NLLLoss()

    # Set up plotting function to display loss / accuracy
    _, ax1 = plt.subplots()
    ax1.set(xlabel='Epoch', ylabel='loss', title='Loss and accuracy')
    ax2 = ax1.twinx()
    ax2.set(ylabel='accuracy')
    ax1.grid()
    plt.ion()

    print(f'Training using {device} started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    for epoch in range(epoch, epochs):
        running_loss = 0
        batch = 0
        for inputs, labels in train_loader:
            batch += 1

            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Show that training is in progress
            print(f'Epoch{epoch+1} TrainingBatch{batch} @{datetime.now().strftime("%H:%M:%S")} loss.item: {loss.item():.3f}')

        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            validation_batch = 0
            for inputs, labels in valid_loader:
                validation_batch += 1

                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                valid_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f'E{epoch+1} ValidationBatch{validation_batch} @{datetime.now().strftime("%H:%M:%S")} loss.item: {batch_loss.item():.3f}')

        print(f'Epoch {epoch+1}/{epochs}.. '
              f'Train loss: {running_loss/batch:.3f}.. '
              f'Validation loss: {valid_loss/validation_batch:.3f}.. '
              f'Validation accuracy: {accuracy * 100 / validation_batch:.3f}%')

        train_loss.append(running_loss/batch)
        validation_loss.append(valid_loss/validation_batch)
        validation_accuracy.append(accuracy * 100/validation_batch)
        ax1.plot(train_loss[2:], color='blue', label='Training loss' if epoch == 0 else '')
        ax1.plot(validation_loss[2:], color='red', label='Validation loss' if epoch == 0 else '')
        ax2.plot(validation_accuracy[2:], color='green', label='Validation accuracy %' if epoch == 0 else '')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.pause(0.1)

        model.train()

        save_checkpoint(model, model_name, optimizer, epoch, train_loss, validation_loss, validation_accuracy, f'{checkpoint_base}{epoch+1}.pth')

    print(f'Training completed, {epoch} epochs, at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

def get_cat_to_name():
    """Load category labels"""
    with open('/home/andrew/Udacity/AI Programming with Python/FlowersClassification/flowers/cat_to_name.json', 'r') as f:
        raw_file = json.load(f)
    print(f'Number of flower labels: {len(raw_file)}')
    return raw_file

def save_checkpoint(model, model_name, optimizer, epoch, train_loss, validation_loss, validation_accuracy, filepath):
    # Save a checkpoint
    checkpoint = {'state_dict': model.state_dict(),
                  'model_name': model_name,
                  'class_to_idx': train_data.class_to_idx,
                  'opt_state': optimizer.state_dict(),
                  'epoch': epoch,
                  'train_loss': train_loss,
                  'validation_loss': validation_loss,
                  'validation_accuracy': validation_accuracy}
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, learning_rate=0.003, hidden_units=4096):
    checkpoint = torch.load(filepath)
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    else:
        model_name = 'vgg11'

    # Create model
    model, optimizer = create_model(model_name, lr=learning_rate, hu=hidden_units)
    # Load model state
    model.load_state_dict(checkpoint['state_dict'])
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['opt_state'])
    # Load class_to_idx mapping
    train_data.class_to_idx = checkpoint['class_to_idx']
    # Epoch - resume training from next epoch
    epoch = checkpoint['epoch'] + 1
    # Training loss curve
    train_loss = checkpoint['train_loss']
    # Validation loss curve
    validation_loss = checkpoint['validation_loss']
    # Validation accuracy curve
    validation_accuracy = checkpoint['validation_accuracy']

    return model, model_name, optimizer, epoch, train_loss, validation_loss, validation_accuracy


def process_image(img_file):
    img_np = Image.open(img_file)
    img_tensor = test_transforms(img_np)
    return img_tensor


def imshow(img_file, probabilities, indices, cat_to_name, label):
    """Show image and predictions"""
    img = mpimg.imread(img_file)

    # Create axes to show image and predictions
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    plt.title(label)
    ax1.imshow(img)
    idx_to_class = dict((v, k) for k, v in train_data.class_to_idx.items())
    names = [f'{cat_to_name[idx_to_class[i]]}' for i in indices]
    y_ticks = np.arange(5)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(names)
    ax2.barh(y_ticks, probabilities)
    plt.tight_layout()
    plt.show()
    plt.close()


def predict(img_file, model, topk):
    """Predict the class (or classes) of an image using a trained deep learning model"""
    img = process_image(img_file)
    img = img.float().unsqueeze_(0)

    with torch.no_grad():
        model.eval()
        output = model.forward(img)
        ps = F.softmax(output.data, dim=1)
        probability, index = ps.topk(topk)

    return probability.reshape(-1).numpy(), index.reshape(-1).numpy()


def get_one_of_each_flower():
    root = 'flowers/test/'
    ret = {}
    for folder in os.listdir(root):
        contents = os.listdir(root + folder)
        if len(contents) > 0:
            ret[folder] = root + folder + "/" + contents[0]

    return ret


def infer(checkpoint, img_files, topk):
    model, model_name, optmizer, epoch, train_loss, validation_loss, validation_accuracy = load_checkpoint(checkpoint)
    cat_to_name = get_cat_to_name()
    for label in img_files:
        probabilities, indices = predict(img_files[label], model, topk)
        imshow(img_files[label], probabilities, indices, cat_to_name, cat_to_name[label])


parser = argparse.ArgumentParser("train from checkpoint")
parser.add_argument('--checkpoint', default='checkpoints/checkpoint20.pth', help='file name (including path) to checkpoint')
parser.add_argument('--infer_image', help='file name (including path) to image for inference')
parser.add_argument('-infer_one_of_each', default=True, action='store_true')
args = parser.parse_args()

if args.infer_image:
    label = args.infer_image.split('/')[-2]
    infer(args.checkpoint, {label: args.infer_image})

elif args.infer_one_of_each:
    img_files = get_one_of_each_flower()
    infer(args.checkpoint, img_files)

