import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import PIL
from PIL import Image
from matplotlib.pyplot import MultipleLocator
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from net import TumorNet
from dataLoader import TumorImageDataset
import nvflare.client as flare


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
print(PIL.__version__)

#img_dir = os.path.realpath('dataset')  # path of image directory
DATASET_PATH = "/Users/leo/Desktop/Praktikum/Repo/NVFlare/dataset"
img_dir = DATASET_PATH
images = os.listdir(img_dir)

PATH = "/Users/leo/Desktop/Praktikum/Repo/NVFlare/TumorNet.pth"



# implement tumor_df as pd.DataFrame for train/test split and further data process
tumor_df = None
img_names, img_labels = zip(*[(i, 0 if 'Not Cancer' in i else 1) for i in images])
names = pd.Series(img_names, name='name')
labels = pd.Series(img_labels, name='label')
tumor_df = pd.concat([names,labels],axis=1)

def main():

    # training set/test set split
    train_set, test_set = train_test_split(tumor_df, train_size=0.8, test_size=0.2, random_state=0)


    # here we resize and cut the center of each image to obtain a dataset with uniform size
    image_transform = transforms.Compose([
        transforms.Resize(size = (256, 256)),
        transforms.CenterCrop(size = (244, 244)),
        transforms.ToTensor()
    ])

    # implement Dataset and DataLoader for training
    train_data = TumorImageDataset(train_set, img_dir, image_transform)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)

    test_data = TumorImageDataset(test_set, img_dir, image_transform)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    image_datasets = {'train': train_data, 'test': test_data}
    image_dataloaders = {'train': train_dataloader, 'test': test_dataloader}

    net = TumorNet()

    flare.init()

    @flare.train
    def train_model(model, loss_func, optimizer, epochs, image_datasets, image_dataloaders):
        model = net        # (optional) use GPU to speed things up
        model.to(device)

        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        for e in range(1, epochs + 1):
            print('Epoch {}/{}'.format(e, epochs))
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # set model to training mode for training phase
                else:
                    model.eval()  # set model to evaluation mode for test phase

                running_loss = 0.0  # record the training/test loss for each epoch
                running_corrects = 0  # record the number of correct predicts by the model for each epoch

                for features, labels in image_dataloaders[phase]:
                    # send data to gpu if possible
                    features = features.to(device)
                    labels = labels.to(device)

                    # reset the parameter gradients after each batch to avoid double-counting
                    optimizer.zero_grad()

                    # forward pass
                    # set parameters to be trainable only at training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outcomes = model(features)
                        pred_labels = outcomes.round()  # round up forward outcomes to get predicted labels
                        labels = labels.unsqueeze(1).type(torch.float)
                        loss = loss_func(outcomes, labels)  # calculate loss

                        # backpropagation only for training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # record loss and correct predicts of each bach
                    running_loss += loss.item() * features.size(0)
                    running_corrects += torch.sum(pred_labels == labels.data)

                # record loss and correct predicts of each epoch and stored in history
                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                history[phase + '_loss'].append(epoch_loss)
                history[phase + '_acc'].append(epoch_acc)

                
        torch.save(model.state_dict(), PATH)
        output_model = flare.FLModel(params=net.cpu().state_dict(), meta={"NUM_STEPS_CURRENT_ROUND": 15})
        return output_model
        

    @flare.evaluate
    def fl_evaluate(model=None):
        return evaluate(input_weights=model.params)
    
    def evaluate(input_weights):

        net.load_state_dict(input_weights)
        # (optional) use GPU to speed things up
        net.to(device)

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_dataloader:
                # (optional) use GPU to speed things up
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(total)
        print(correct)
        print(f"Accuracy of the network on the 480 test images: {100 * correct // total} %")
        return 100 * correct // total
    
 

    while flare.is_running():
        # (6) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # (7) call fl_evaluate method before training
        #       to evaluate on the received/aggregated model
        global_metric = fl_evaluate(input_model)
        print(f"Accuracy of the global model on the 480 test images: {global_metric} %")
        # call train method
        train_model(
        model = net,
        loss_func=nn.BCELoss(),
        optimizer=optim.Adam(net.parameters(), lr=0.001),
        epochs=15,
        image_datasets=image_datasets,
        image_dataloaders=image_dataloaders
        )
        # call evaluate method
        metric = evaluate(input_weights=torch.load(PATH))
        print(f"Accuracy of the trained model on the 10000 test images: {metric} %")
    

if __name__ == "__main__":
    main()