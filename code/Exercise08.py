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
import nvflare.client as flare


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
print(PIL.__version__)

do_training = True

#img_dir = os.path.realpath('dataset')  # path of image directory
DATASET_PATH = "/Users/leo/Desktop/Praktikum/Repo/NVFlare/code/dataset"
img_dir = DATASET_PATH
images = os.listdir(img_dir)

PATH = "/Users/leo/Desktop/Praktikum/Repo/NVFlare/TumorNet.pth"



# implement tumor_df as pd.DataFrame for train/test split and further data process
tumor_df = None
img_names, img_labels = zip(*[(i, 0 if 'Not Cancer' in i else 1) for i in images])
names = pd.Series(img_names, name='name')
labels = pd.Series(img_labels, name='label')
tumor_df = pd.concat([names,labels],axis=1)


assert len(tumor_df) == 460, 'Please check the set up of tumor_df'
assert tumor_df["label"].value_counts()[1] == 252
assert tumor_df["label"].value_counts()[0] == 208

def main():


    # training set/test set split
    train_set, test_set = train_test_split(tumor_df, train_size=0.8, test_size=0.2, random_state=0)

    class TumorImageDataset(Dataset):
        """load, transform and return image and label"""

        def __init__(self, annotations_df, img_dir, transform=None):
            self.img_labels = annotations_df
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            # get image path according to idx
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            # convert all image to RGB format
            image = Image.open(img_path).convert('RGB')
            label = self.img_labels.iloc[idx, 1]
            # apply image transform
            if self.transform:
                image = self.transform(image)
            return [image, label]


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





    # Unfortunately, Pytorch does not support the build-in method to display model information (such as the `summary()` method in Tensorflow). Luckily this can be easily implemented. Here is an example:
    def summary(model):
        """Print out model architecture infomation"""
        parameter_count = 0
        model_info = model.state_dict()
        for name, module in model.named_children():
            # loop each module in the model to record number of parameters
            try:
                n_weight = model_info[name + '.weight'].flatten().shape[0]
                n_bias = model_info[name + '.bias'].flatten().shape[0]
            except:
                n_weight = 0
                n_bias = 0
            print(f'{name} layer (No. of weight: {n_weight:n}, No. of bias: {n_bias:n})')
            parameter_count += (n_weight + n_bias)
        print(f'Total parameters: {parameter_count:n}')



    net = TumorNet()


    flare.init()


    @flare.train
    def train(input_model, loss_func, optimizer, epochs, image_datasets, image_dataloaders, do_training=True):
        """Return the trained model and train/test accuracy/loss"""
        

        net.load_state_dict(input_model.params)

        net.to(device)

        for e in range(1, epochs + 1):
            print('Epoch {}/{}'.format(e, epochs))
            for phase in ['train', 'test']:
                if phase == 'train':
                    input_model.train()  # set model to training mode for training phase
                else:
                    input_model.eval()  # set model to evaluation mode for test phase

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
                        outcomes = input_model(features)
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
                torch.save(input_model.state_dict(), PATH)


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
        train(
        input_model,
        loss_func=nn.BCELoss(),
        optimizer=optim.Adam(net.parameters(), lr=0.001),
        epochs=15,
        image_datasets=image_datasets,
        image_dataloaders=image_dataloaders,
        do_training=do_training
        )
        # call evaluate method
        metric = evaluate(input_weights=torch.load(PATH))
        print(f"Accuracy of the trained model on the 10000 test images: {metric} %")
    

if __name__ == "__main__":
    main()
