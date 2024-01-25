import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from net import TumorNet
from dataLoader import TumorImageDataset, dataLoader
from prepare_data_split import createSplit
import nvflare.client as flare

from nvflare.client.tracking import MLflowWriter

from nvflare.app_opt.pt.scaffold import PTScaffoldHelper, get_lr_values
from nvflare.app_common.app_constant import AlgorithmConstants




device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
print(PIL.__version__)

#img_dir = os.path.realpath('dataset')  # path of image directory
DATASET_PATH = "/Users/leo/Desktop/Praktikum/Repo/NVFlare/dataset"
img_dir = DATASET_PATH
images = os.listdir(img_dir)

NET_PATH = "/Users/leo/Desktop/Praktikum/Repo/NVFlare/TumorNet.pth"
OUTPUT_PATH = "../output"

def main(batch_sz, epochs, lr, split_method):
    #create json file of the data split and save it within OUTPUT_PATH
    jsonSplit = createSplit(data_path = DATASET_PATH,
            site_num = 2,
            site_name_prefix = "site-",
            split_method = split_method,
            out_path = OUTPUT_PATH
            )
    

    net = TumorNet()

    flare.init()
    client_id = flare.get_site_name() 

    scaffold_helper = PTScaffoldHelper()
    scaffold_helper.init(model = net)






    writer = MLflowWriter()

    params={'batch_size': batch_sz, 
            'epoch': epochs,
            'learning_rate': lr,
            'split_method': split_method}
    
    writer.log_params(params)

    splitted_data = dataLoader.load_data(jsonSplit, client_id)
    
    tumor_df_split = splitted_data[0]
    valid_set = splitted_data[1]

    # training set/test set split
    train_set, test_set = train_test_split(tumor_df_split, train_size=0.8, test_size=0.2, random_state=0)


    # here we resize and cut the center of each image to obtain a dataset with uniform size
    image_transform = transforms.Compose([
        transforms.Resize(size = (256, 256)),
        transforms.CenterCrop(size = (244, 244)),
        transforms.ToTensor()
    ])

    
    
    # implement Dataset and DataLoader for training, testing and validation
    train_data = TumorImageDataset(train_set, img_dir, image_transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_sz, shuffle=False)

    test_data = TumorImageDataset(test_set, img_dir, image_transform)
    test_dataloader = DataLoader(test_data, batch_size=batch_sz, shuffle=False)

    valid_data = TumorImageDataset(valid_set, img_dir, image_transform)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_sz, shuffle=False)

    image_datasets = {'train': train_data, 'test': test_data, 'valid': valid_data}
    image_dataloaders = {'train': train_dataloader, 'test': test_dataloader, 'valid': valid_dataloader}

    
    @flare.train
    def train_model(input_model, loss_func, optimizer, epochs, image_datasets, image_dataloaders):
        net.load_state_dict(input_model.params)

        c_global_para, c_local_para = scaffold_helper.get_params()

        model_global = copy.deepcopy(net)

        # (optional) use GPU to speed things up
        net.to(device)

        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        for e in range(1, epochs + 1):
            print('Epoch {}/{}'.format(e, epochs))
            for phase in ['train', 'test']:
                if phase == 'train':
                    net.train()  # set model to training mode for training phase
                else:
                    net.eval()  # set model to evaluation mode for test phase

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
                        outcomes = net(features)
                        pred_labels = outcomes.round()  # round up forward outcomes to get predicted labels
                        labels = labels.unsqueeze(1).type(torch.float)
                        loss = loss_func(outcomes, labels)  # calculate loss

                        # backpropagation only for training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        curr_lr = get_lr_values(optimizer)[0]
                        
                        scaffold_helper.model_update(
                            model=net, curr_lr=curr_lr, c_global_para=c_global_para,
                              c_local_para=c_local_para
                        )



                    # record loss and correct predicts of each bach
                    running_loss += loss.item() * features.size(0)
                    running_corrects += torch.sum(pred_labels == labels.data)

                # record loss and correct predicts of each epoch and stored in history
                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])

                metrics={f'epoch_loss_{phase}':epoch_loss, f'epoch_acc_{phase}':epoch_acc.item()}
                #print(f'log the {phase} accuracy for epoch {e}')
                
                writer.log_metrics(
                metrics=metrics,
                step= e + input_model.current_round
                )
                
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                history[phase + '_loss'].append(epoch_loss)
                history[phase + '_acc'].append(epoch_acc)
        
        print(f"Finished Training of {client_id} \n")

        torch.save(net.state_dict(), NET_PATH)


        scaffold_helper.terms_update(
                model=net,
                curr_lr=curr_lr,
                c_global_para=c_global_para,
                c_local_para=c_local_para,
                model_global=model_global,
            )
        

        metric = evaluate(input_weights=torch.load(NET_PATH))
        print(
            f"Accuracy of the trained model on the test images: {metric} %"
        )
        



        output_model = flare.FLModel(params=net.cpu().state_dict(),
                                     optimizer_params= optimizer.state_dict(),
                                     metrics={"accuracy": metric},
                                      meta={
            AlgorithmConstants.SCAFFOLD_CTRL_DIFF: scaffold_helper.get_delta_controls(),
     
        },)
        
        
        
        
        return output_model
        

    @flare.evaluate
    def fl_evaluate(model=None):
        return evaluate(input_weights=model.params)
    
    def evaluate(input_weights):
         
        net.load_state_dict(input_weights)
        # (optional) use GPU to speed things up
        net.to(device)

        net.eval()  # set model to evaluation mode for test phase

        correct = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for features, labels in image_dataloaders['valid']:

                # send data to gpu if possible
                features = features.to(device)
                labels = labels.to(device)

                # calculate outputs by running images through the network
                with torch.set_grad_enabled(False):
                    outputs = net(features)
                    pred_labels = outputs.round()  # round up forward outcomes to get predicted labels
                    labels = labels.unsqueeze(1).type(torch.float)
                    
                    correct += (pred_labels == labels).sum().item()
                
        return 100 * correct // len(image_datasets['valid'])
    

    while flare.is_running():
        # receive FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # call fl_evaluate method before training to evaluate on the received/aggregated model
        global_metric = fl_evaluate(input_model)
        print(f"Accuracy before training on {len(image_datasets['valid'])} test images: {global_metric}%  on {client_id} \n")
        
        #(7) if you want to LOG the global metric
        writer.log_metric('global_accuracy_after_ EACH_RUN', global_metric)
        

        if AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL not in input_model.meta:
            raise ValueError(
                f"Expected model meta to contain AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL "
                f"but meta was {input_model.meta}.",
            )
        
        
        global_ctrl_weights = input_model.meta.get(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL)
        if not global_ctrl_weights:
            raise ValueError("global_ctrl_weights were empty!")
        

        for k in global_ctrl_weights.keys():
            global_ctrl_weights[k] = torch.as_tensor(global_ctrl_weights[k])
        

        scaffold_helper.load_global_controls(weights=global_ctrl_weights)





        # call train method
        train_model(
        input_model = input_model,
        loss_func=nn.BCELoss(),
        optimizer=optim.Adam(net.parameters(), lr=lr),
        epochs=epochs,
        image_datasets=image_datasets,
        image_dataloaders=image_dataloaders
        )

        sys_info = flare.system_info()
        # LOG SOME INFO ABOUT THE CLIENT IN THE MLFLOW AS TAGS 
        writer.set_tags(sys_info)
        # call evaluate method
        
        metric = evaluate(input_weights=torch.load(NET_PATH))
        print(f"Accuracy after training on {len(image_datasets['valid'])} test images: {metric}% on {client_id} \n")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a model for brain tumor detection using nvflare."
    )

    # Add an argument for batch_sz
    parser.add_argument(
        "--batch_sz",
        type=int,
        default=None,
        help="Specify the batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None, 
        help="learning rate"
    )

    parser.add_argument(
        "--split_method",
        type=str, 
        default="uniform",
        choices=["uniform", "linear", "square", "exponential"],
        help="How to split the dataset",
    )

    args = parser.parse_args()
    main(batch_sz=args.batch_sz, epochs=args.epochs, lr=args.lr, split_method=args.split_method)