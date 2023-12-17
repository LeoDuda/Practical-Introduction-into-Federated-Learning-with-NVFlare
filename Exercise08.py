# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "73981ab9c0be5ca69e8dc1aac9b17034", "grade": false, "grade_id": "cell-1da0d1c590ff8a40", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Exercise Sheet No. 8
#
# ---
#
# > Machine Learning for Natural Sciences, Summer 2023, Jun.-Prof. Pascal Friederich, pascal.friederich@kit.edu
# > 
# > Deadline: June 19th 2023, 8:00 am
# >
# > Container version 1.0.2
# >
# > Tutor: patrick.reiser@kit.edu
# >
# > **Please ask questions in the forum/discussion board and only contact the Tutor when there are issues with the grading**
#
# ---
#
# **Topic**: This assignment will focus on convolutional neural network for brain tumor image classification.

# %% [markdown]
# Please add here your group members' names and student IDs. 
#
# Names: Leonhard Duda
#
# IDs: 1939668

# %% [markdown]
# # Preliminaries
# In this assginment, we are going to use Pytorch instead of Tensorflow to build our neural network.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "ccce8649e39c374be69b710e5d31b341", "grade": false, "grade_id": "cell-18eae86a8da21038", "locked": true, "schema_version": 3, "solution": false, "task": false}
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
print(PIL.__version__)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "bef19bdc929746b9518c2ac8e972fbe3", "grade": false, "grade_id": "cell-49cf292b54724ea2", "locked": true, "schema_version": 3, "solution": false, "task": false}
# For the purpose of autograding, please:
# 1. Set `do_training=True` while finishing this assignment.
# 2. Please submit your solution with `do_training = False`.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "8f77ae003831fc668c4f278deffb72a6", "grade": false, "grade_id": "cell-ded537f716d4ca11", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Please submit your solution with do_training = False.
do_training = True

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0ae0b709882b04c6434e1289055e11cd", "grade": false, "grade_id": "cell-95457f6c58e25bf5", "locked": true, "schema_version": 3, "solution": false, "task": false}
# In the last assignments, we have implemented and trained fully connected neural networks with applications in chemistry and materials science. This time, we will learn to implement and train a convolutional neural network (CNN) for brain tumor image classification, and compare it with a fully connected neural network to show the power of convolutional filters. In addition to that, we will see how a pretrained network can be used to fulfill the same task with better performance through transfer learning.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "c439be72e91984f760d356fa04d1534f", "grade": false, "grade_id": "cell-cb8513bebe730708", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Dataset
# In this assignment, we will work with images of brain x-rays from both healthy people and cancer patients.
#
# A brain tumor occurs when abnormal cells form within the brain for unknown reasons. And primary brain tumors occur in around 250,000 people a year globally, making up less than 2% of cancers. The early diagnosis is important when fighting this disease, and medical imaging plays a central role in this field. You may read more about brain tumor from this [paper](https://pubmed.ncbi.nlm.nih.gov/27157931/).
#
# This data set was downloaded from [kaggle](https://www.kaggle.com/preetviradiya/brian-tumor-dataset), and only 10% of cancer/not cancer images were selected randomly for this assignment (460 images in total). Now, let's get started with visualizing part of the dataset.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "bad951963ce52a81de75484af451feaf", "grade": false, "grade_id": "cell-f6dde6886883799c", "locked": true, "schema_version": 3, "solution": false, "task": false}
img_dir = os.path.realpath('dataset')  # path of image directory
images = os.listdir(img_dir)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "ecaf855fa877962a44e7888d21551aa3", "grade": false, "grade_id": "cell-927bc59378df48fb", "locked": true, "schema_version": 3, "solution": false, "task": false}
# plot 10 x-ray images
fig, axes = plt.subplots(2, 5, figsize=(40, 20), gridspec_kw=dict(hspace=0.1, wspace=0.3))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(Image.open(os.path.join(img_dir, images[i])))
    ax.set_title(images[i])

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5ff772a6170a2bd3928cd3f866898262", "grade": false, "grade_id": "cell-e03ff8d281630822", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Principal Component Analysis (PCA)
# From the plot above we can see that images in the dataset vary by several features, such as brightness, orientation, and of course whether they have a tumor or not. For such a complicated dataset, is it possible to classify it with a simple classifier like k-nearest neighbors? Principal Component Analysis can give us a hint on this problem.
#
# PCA is an unsupervised technique to project the data to lower dimensional space through linear (linear PCA) or non-linear (kernel PCA) combinations of the data's original features. The "Principal Component" stands for combination results of features that account for most of a dataset's variance. PCA can also be used to visualize high-dimensional data, filter out noise, or as feature representation. Here, we will take this advantage to visualize the most significant features in our dataset.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "064b33cc28deb225a6e8496ec60e29a1", "grade": false, "grade_id": "cell-87fb0bcc79cd5157", "locked": true, "schema_version": 3, "solution": false, "task": false}
# prepare data in numpy array for PCA()
data = [Image.open(os.path.join(img_dir, i)).convert('L').resize((256, 256)) for i in images]
data = np.array([np.array(d).reshape(-1) for d in data])

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "9c5d6ecc537aa7aeb43d75de8eeaa657", "grade": false, "grade_id": "cell-03eafe24d8cff8e5", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Please use the `PCA()` to perform principal component analysis on `data`. Use the parameter `n_components` to set the amount of variance that needs to be explained to be 0.8.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "93b3d0526687643fb52f31b6b8d0369a", "grade": false, "grade_id": "cell-c58aaa5b6168268d", "locked": false, "schema_version": 3, "solution": true, "task": false}

pca = PCA(n_components = 0.8)
pca.fit(data)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "a19f9867c1c2fd45d07a0c7c0fbbe3cc", "grade": true, "grade_id": "pca", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
assert isinstance(pca, PCA), "pca should be an instance of the sklearn PCA"

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0d7831a3c3dc77831ad490b675e59c7d", "grade": false, "grade_id": "cell-3630eebe8ffff6ff", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now, let's visualize the images associated with the first several principal components, which may give us some insight about what make these images vary from each other.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "9dc67cab2a64171a3b2d63eaf99b6cc5", "grade": false, "grade_id": "cell-455941975389eb8e", "locked": true, "schema_version": 3, "solution": false, "task": false}
# visualize the first 10 principal components
fig, axes = plt.subplots(2, 5, figsize=(40, 20), gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(pca.components_[i].reshape(256, 256), cmap='gray')

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "034f6e22b94edc0dc68c9f977a003cbd", "grade": false, "grade_id": "cell-e0652fd2149d41c2", "locked": true, "schema_version": 3, "solution": false, "task": false}
# As can be seen above, the first two images on the top left seem to be associated with the brightness of the x-ray image, while the later ones on the bottom right seem related to the direction or size of the skull. Since none of them displays strong relationship to the shape/size/location of the brain tumor, it is difficult to classify cancer/not cancer images with a simple algorithm, which may focus more on those more 'obvious' features. As a result, a more sophisticated model like a neural network is more suitable for this task.

# %% [markdown]
# ## Data processing
# To begin with, we need to obtain and organize all image metadata with `pandas`. Please use `pd.Series` for `img_names` and `img_labels`, together with `pd.concat()` and `pd.DataFrame` to construct `tumor_df`.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "e98ef61f16890fc5f7e3e5ec086a6494", "grade": false, "grade_id": "cell-f2038a6ccb8058c8", "locked": false, "schema_version": 3, "solution": true, "task": false}
# implement tumor_df as pd.DataFrame for train/test split and further data process
tumor_df = None
img_names, img_labels = zip(*[(i, 0 if 'Not Cancer' in i else 1) for i in images])
names = pd.Series(img_names, name='name')
labels = pd.Series(img_labels, name='label')
tumor_df = pd.concat([names,labels],axis=1)
display(tumor_df)


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "98a2b8d78a464dedfbdb3a8a84e6b727", "grade": true, "grade_id": "dataframe", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
assert len(tumor_df) == 460, 'Please check the set up of tumor_df'
assert tumor_df["label"].value_counts()[1] == 252
assert tumor_df["label"].value_counts()[0] == 208

# %% [markdown]
# Now, for the training/test set generation, please use the `train_test_split()` to split the `tumor_df` into `train_set` and `test_set`. This time, we will use 80% of the data for training and the rest for testing.
#
# Please fix the `random_state` to 0.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "3c4a55914fb4ba5f8c4be709d3412dea", "grade": false, "grade_id": "cell-e33a37c6ea1a2464", "locked": false, "schema_version": 3, "solution": true, "task": false}
# training set/test set split
train_set, test_set = train_test_split(tumor_df, train_size=0.8, test_size=0.2, random_state=0)


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "077c34012c577047a184ca76b92e27aa", "grade": true, "grade_id": "data-split", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# train-test split 1 point
assert train_set.shape == (368, 2)
assert test_set.shape == (92, 2)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "6d178e1322fe1a352771e8e8666c318f", "grade": false, "grade_id": "cell-9f6286c040c18dd3", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Pytorch offers convenient APIs for handling data: `Dataset` and `DataLoader`, which achieve data loading/processing with better readability and modularity. Take the following implementation as an example. Note that this implementation is only for demonstrational purposes. Since our data fits into memory, a more performant implementation would would use a `TensorDataset` or cache images. Repetitively loading samples from disk is inefficient and slows down training significantly.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "2fc542a3a8b36c58d80806a367f0e77f", "grade": false, "grade_id": "cell-505e3b65ddc27ff5", "locked": true, "schema_version": 3, "solution": false, "task": false}
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


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ec53cac592a71e62f3b6849bec3a50a9", "grade": false, "grade_id": "cell-600ac4a77b30dbf1", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Please finish the `image_transform` module. The image should be resized to $256\times 256$ then cropped at center to size $244 \times 244$. You may use `transforms.Resize()` and `transforms.CenterCrop()` for this task.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "366191b353424c5158137b60ac9973e8", "grade": false, "grade_id": "cell-96a3bdb0a80f45b1", "locked": false, "schema_version": 3, "solution": true, "task": false}
# user defined image transform process
# here we resize and cut the center of each image to obtain a dataset with uniform size
image_transform = transforms.Compose([
    transforms.Resize(size = (256, 256)),
    transforms.CenterCrop(size = (244, 244)),
    transforms.ToTensor()
])

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e7ae43be7ba75ae01bafca652b782b83", "grade": false, "grade_id": "cell-a41ca969e64f7962", "locked": true, "schema_version": 3, "solution": false, "task": false}
# implement Dataset and DataLoader for training
train_data = TumorImageDataset(train_set, img_dir, image_transform)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)

test_data = TumorImageDataset(test_set, img_dir, image_transform)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

image_datasets = {'train': train_data, 'test': test_data}
image_dataloaders = {'train': train_dataloader, 'test': test_dataloader}

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "49ed380629f959106843109a1ddb2663", "grade": true, "grade_id": "data-transform", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
feat, labels = next(iter(train_dataloader))
assert feat.shape[2] == 244 and feat.shape[3] == 244, "Wrong size, please check image_transform"

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "f55b6528631934e9169206566d264363", "grade": false, "grade_id": "cell-aa5c269e894ec6aa", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Here is a plot for part of the dataset with `train_dataloader`.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "7831085a58309a3fb97c0b7d5021f10a", "grade": false, "grade_id": "cell-648363a3d49dc0ca", "locked": true, "schema_version": 3, "solution": false, "task": false}
# plot a batch (32) of image in training set
train_features, train_labels = next(iter(train_dataloader))  # DataLoader is iterable
fig, axes = plt.subplots(4, 8, figsize=(40, 20), gridspec_kw=dict(hspace=0.1, wspace=0.3))
for i, ax in enumerate(axes.flat):
    ax.imshow(train_features[i].numpy().transpose((1, 2, 0)))
    ax.set_title('Cancer' if int(train_labels[i] == 1) else 'Not Cancer')


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "d3b03f50b5f796b5770b7d3f6ba4060c", "grade": false, "grade_id": "cell-7e1e3361387e319a", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Build the neural network
# ## motivation of CNN: compare with fully connected layers
# As discussed in the class, an advantage of convolutional neural network (CNN) is weight-sharing of filter parameters. Let's demonstrate this effect by looking at a densely connected NN.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "f680e466a8a38f4a7b3ee897ec56b933", "grade": false, "grade_id": "cell-5832647023fde8ce", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Pytorch builds neural network by subclassing the `nn.Module`. For each `nn.Module`, there is a `forward()` method implementing operations on input data.
#
# We implement the `MultiLayerPerceptron` with three fully connected layers. Please finish the `forward()` method function using `self.fc1`, `self.fc2`, `self.fc3` that have already been already defined. For the first two layers, there is a `F.relu()` activation function for each of them. For the output layer, we use `torch.sigmoid()` to generate outcome between 0 and 1.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "22fcb54a73d23fe72af2c026086761e9", "grade": false, "grade_id": "cell-16b77cd218f29cfb", "locked": false, "schema_version": 3, "solution": true, "task": false}
class MultiLayerPerceptron(nn.Module):
    """A three layer fully connected neural network"""

    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(in_features=244 * 244 * 3, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        """Operations on x"""
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "fe11a1f00e7b388afc94f2a7c2d6b8ab", "grade": true, "grade_id": "mlp-train", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false}
mlp_model = MultiLayerPerceptron().to(device)

assert len(mlp_model.state_dict().keys()) == 6, 'please check if there is any fc layer mising in forward()'


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "93dc0b9cce1ed469e7e556abf06ecfcb", "grade": false, "grade_id": "cell-e0b232bcf59dea20", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Unfortunately, Pytorch does not support the build-in method to display model information (such as the `summary()` method in Tensorflow). Luckily this can be easily implemented. Here is an example:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "62820375a6644e52824dd7880feb940a", "grade": false, "grade_id": "cell-b6474c1d775dd8b8", "locked": true, "schema_version": 3, "solution": false, "task": false}
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


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "80c5ab64fb991a90c006d7c636cab1f5", "grade": false, "grade_id": "cell-84cc8ce966152000", "locked": true, "schema_version": 3, "solution": false, "task": false}
summary(mlp_model)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "46af38eb8970cd53e179ad018a372b9d", "grade": false, "grade_id": "cell-224de6ad1f61687b", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Training
# As seen from the `summary()` result, dense networks dealing with images can become very large (more than $1e7$ parameters in this case). This can lead to problems such as long training time. Please try to train this network for 1 epoch to get an idea about it, using the `train_model()` function defined below.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "a4529647730b5b3034a146d27d8e2df2", "grade": false, "grade_id": "cell-d9019b4a6f0307d7", "locked": true, "schema_version": 3, "solution": false, "task": false}
def train_model(model, loss_func, optimizer, epochs, image_datasets, image_dataloaders, do_training=True):
    """Return the trained model and train/test accuracy/loss"""
    if not do_training:
        return None, None
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

    return model, history


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "3707482423fe735cf32b74d6e148f7bf", "grade": false, "grade_id": "cell-54541eb2035ac424", "locked": true, "schema_version": 3, "solution": false, "task": false}
# For binary classification, we will use Binary Cross Entropy `BCELoss()` as loss function (read more information in [documentation](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)). We use `optim.Adam()` for optimization. For a more detailed explanation, you may read the original paper from 2015 [here](https://arxiv.org/abs/1412.6980) .

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "1bf3c93732a9d3795d5c9131c962e15e", "grade": false, "grade_id": "cell-a37cc3412f543c3a", "locked": true, "schema_version": 3, "solution": false, "task": false}
mlp_model_trained, history = train_model(
    model=mlp_model,
    loss_func=nn.BCELoss(),
    optimizer=optim.Adam(mlp_model.parameters(), lr=0.001),
    epochs=1,
    image_datasets=image_datasets,
    image_dataloaders=image_dataloaders,
    do_training=do_training
)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "aabb30803a434419fcff9fe1f2dd01ef", "grade": false, "grade_id": "cell-6277c97224d0cc41", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Convolutional Neural Network 
# Now, let's build our own CNN `TumorNet`. Please finish the convolutional part in `forward()`:
# 1. Use `self.conv` as the convolutional layer with the `F.relu()` activation function.
# 2. Add max pooling layer `self.pool`. 
# 3. The feature map is then flattened and feed into `self.fc1` with `F.relu()` activation function.
# 4. We will then use `self.dropout` for regularization.
# 5. Use `self.fc2` as output layer with `torch.sigmoid()` as activation function.
#
# ![CNN](./img/TumorNet.png)

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "6876f0abbd442e4e4559adeda253395e", "grade": false, "grade_id": "cell-b45ecccf6a9e4b09", "locked": false, "schema_version": 3, "solution": true, "task": false}
class TumorNet(nn.Module):
    """
    A CNN with:
        one convolutional layer
        one max pooling layer
        two fully connected layers
    """

    def __init__(self):
        super(TumorNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=60 * 60 * 16, out_features=64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        """Operations on x"""
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "5fdb752e9a492c3dd1fbb8e115fe01ec", "grade": true, "grade_id": "tumornet-train", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false}
cnn_model = TumorNet().to(device)
summary(cnn_model)

assert 'conv.weight' in cnn_model.state_dict().keys()


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "5bd8ea3da40f951f03d3d8b3f08957df", "grade": false, "grade_id": "cell-e14cffd9d3add841", "locked": true, "schema_version": 3, "solution": false, "task": false}
cnn_model_trained, cnn_history = train_model(
    model=cnn_model,
    loss_func=nn.BCELoss(),
    optimizer=optim.Adam(cnn_model.parameters(), lr=0.001),
    epochs=15,
    image_datasets=image_datasets,
    image_dataloaders=image_dataloaders,
    do_training=do_training
)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "d9adca45f2f2ee1fdb6917260db29b63", "grade": false, "grade_id": "cell-ffc79766fd81a70d", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Please enter your final **test accuracy** after 15 epochs of training.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "6bb8322e58ded67de38a3cba76ceeeab", "grade": false, "grade_id": "cell-482e122509acb638", "locked": false, "schema_version": 3, "solution": true, "task": false}
test_acc = (0.8043)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "b1f1affaaf959bab7f4dbbb6946ef76b", "grade": true, "grade_id": "acc-test-1", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
assert 0 <= test_acc <= 1


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "056d4421d3c5e1a92b479a8c14284af7", "grade": false, "grade_id": "cell-4f16f722063f6aaf", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now we plot the training curve to visualize the loss and accuracy vs. epoch for both training and test process.
# Note that the hyperparameters of this model are not optimized yet. Feel free to give a try on hyperparameter optimization for better result. But please do it in another notebook.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "bb83d934dea56ded02b81f75e4cb2bfb", "grade": false, "grade_id": "cell-411c156c6b081bdf", "locked": true, "schema_version": 3, "solution": false, "task": false}
def plot_training_curve(history):
    """Plot the training curve"""
    train_loss = history['train_loss']
    test_loss = history['test_loss']
    train_acc = history['train_acc']
    test_acc = history['test_acc']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    ax1.plot(list(range(1, len(train_loss) + 1)), train_loss, label='Training', color='c')
    ax1.plot(list(range(1, len(train_loss) + 1)), test_loss, label='Test', color='b')
    x_major_locator = MultipleLocator(1)
    ax1.set_xlim(1, len(train_loss))
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.set_xlabel('Eopchs')
    ax1.set_ylabel('Binary Cross Entropy Loss')
    ax1.legend(loc='upper right', fontsize='x-large')
    ax1.set_title('Loss vs. Epochs')

    ax2.plot(np.arange(1, len(train_acc) + 1), train_acc, label='Training', color='c')
    ax2.plot(np.arange(1, len(train_acc) + 1), test_acc, label='Test', color='b')
    x_major_locator = MultipleLocator(1)
    ax2.set_xlim(1, len(train_acc))
    ax2.xaxis.set_major_locator(x_major_locator)
    ax2.set_xlabel('Eopchs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right', fontsize='x-large')
    ax2.set_title('Accuracy vs. Epochs')

    plt.show()
    plt.close()


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "4bd2d79cb7511aa1cc15134fd92ddc6d", "grade": false, "grade_id": "cell-4afaabad5f66ad0c", "locked": true, "schema_version": 3, "solution": false, "task": false}
try:
    plot_training_curve(cnn_history)
except:
    pass

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "749b53c106d8d8a394345b8f04a831fe", "grade": false, "grade_id": "cell-7c8de36b4d0bcaa6", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Transfer learning
# In practice, people tend to start with pretrained models instead of training an entire Convolutional Network from scratch. The idea is to take the advantage of an existing model, which is trained on a very large dataset, through transfer learning to achieve both better performance and faster convergence. This is particularly useful when training with a dataset of which the size is not sufficient, such as the case we have in this assignment (You may have already noticed the lack of stability during the training of `cnn_model` earlier).

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "fc064b4a9a980a92c38b315155457534", "grade": false, "grade_id": "cell-232ca24810169c37", "locked": true, "schema_version": 3, "solution": false, "task": false}
# There are two major scenarios where transfer learning is used:
# 1. Finetuning the CNN: Instead of random initialization, the network was first initialized with a pretrained network, then trained on the target dataset. In this scenario, parameters in both convolutional and fully connected layers are trainable.
# 2. Used as feature extractor: In this case, all layers except the last fully connected layer were used as feature extractor of which parameters are freezed. The last fc layer is replaced in accordance with the target dataset and its parameters are trainable.
#
# For more infomation about transfer learning, take a look at the [note](https://cs231n.github.io/transfer-learning/).

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ba037bf2d004c5ddca04a314c7fb2182", "grade": false, "grade_id": "cell-13dd8dfd21f3640c", "locked": true, "schema_version": 3, "solution": false, "task": false}
# In this assignment, we are going to take the second choice and use ResNet18 as our pretrained feature extractor. 
# ResNet, or residual network, is a neural network architecture that applies identity mapping as a "shortcut connections" (see figure below). This architecture overcomes the problem of "vanishing gradients" of deep neural network that causes the decrease of prediction accuracy when number of layer increase. For more information, please read the original [paper](https://arxiv.org/abs/1512.03385).
#
# ![resnet](./img/ResNet.png)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "f8821cff08d083dd34062e3a9aff14ce", "grade": false, "grade_id": "cell-c99437ea2bd5aa7f", "locked": true, "schema_version": 3, "solution": false, "task": false}
# To use ResNet18 as a feature extractor, we first need to load the model and freeze all parameters. Please finish the rest steps as described here:
# 1. Replace the last fully connected layer with the desired module for tumor classification task. You may use Pytorch `nn.Sequential` module to concatenate `nn.Linear(in_features, out_features)` and `nn.Sigmoid`. You may use `resnet_model.fc.in_features` to obtain `in_features`. The `out_features` should be 1 since we are working on the binary classification problem with single output node.
# 2. Assign the result Sequential module to the last layer of pretrained model `resnet_model.fc`.
#
# Do not worry about setting the parameters of the last layer to be trainable, as newly constructed modules have `requires_grad=True` by default.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "aef34095092d38a0dbd1f0ac94e72743", "grade": false, "grade_id": "cell-d50994778c5d8796", "locked": false, "schema_version": 3, "solution": true, "task": false}
# load the pretrained ResNet18 model
resnet_model = torchvision.models.resnet18(pretrained=True).to(device)
for p in resnet_model.parameters():
    # freeze all parameters
    p.requires_grad = False
    
in_features = resnet_model.fc.in_features
    
resnet_model.fc = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid()
)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "21a0947ce193eb7b94bd5c33f36cd1c2", "grade": false, "grade_id": "cell-6cba5871e60cd982", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Let's train this model and compare it with the CNN model implemented earlier.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "de7b0ca6ba778c93be29a88e85f482d3", "grade": false, "grade_id": "cell-df333fbc365169a4", "locked": true, "schema_version": 3, "solution": false, "task": false}
resnet_trained, resnet_history = train_model(
    model=resnet_model,
    loss_func=nn.BCELoss(),
    optimizer=optim.Adam(resnet_model.parameters(), lr=0.001),
    epochs=20,
    image_datasets=image_datasets,
    image_dataloaders=image_dataloaders,
    do_training=do_training
)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0814e69dbe4f4a35133d2e59a284d3b8", "grade": false, "grade_id": "cell-1e5fd9f062d794a1", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Please add the final **test accuracy** for the `resnet_model` after 20 epochs of training.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "684baea30514b08d36f4f02a30d786bb", "grade": false, "grade_id": "cell-4e8ac8f2e3743767", "locked": false, "schema_version": 3, "solution": true, "task": false}
test_acc = (
    # YOUR CODE HERE
    raise NotImplementedError()
)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "c1755a7de94754c6e7793645063c8cc5", "grade": true, "grade_id": "acc-test-2", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false}
assert 0 <= test_acc <= 1


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "2ae5d5396bdfee12a5d7adb80307ff15", "grade": false, "grade_id": "cell-e6d82414cf526d64", "locked": true, "schema_version": 3, "solution": false, "task": false}
try:
    plot_training_curve(resnet_history)
except:
    pass

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "7dac1d40953f7cd4a0726929ad8e32c1", "grade": true, "grade_id": "do-training-false", "locked": true, "points": 1, "schema_version": 3, "solution": false, "task": false}
# Please set do_training at the beginning of this assignment to False. Thank you!
assert do_training == False

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8767492ce4c4a353e11e555bafad9e40", "grade": false, "grade_id": "cell-3972d28a22191913", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Hope you enjoy this assignment. Thank you very much!
# **This is the end of the assignment**
