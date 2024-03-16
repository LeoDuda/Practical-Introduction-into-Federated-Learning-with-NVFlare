# Practical-Introduction-into-Federated-Learning-with-NVFlare

This introduction to federated learning with NVFlare is part of the practical course Advanced Topics in High Performance Computing, Data Management and Analytics within the winter semeser 2023/24 at [KIT (Karlsruhe Institute of Technology)](https://www.kit.edu/index.php).
The goal was to understand the theory behind federated learning and used to NVFlare by transforming an existing machine learning workflow into a federated learning paradigm. 

# The use case
As a machine learning use case a brain tumor image classifier was chosen, containing a convolutional neural network with 5 layers and 3.6 million parameters. 
The data consists of 4600 labeled x-ray images. In this use case the CNN was trained such that it can detect a tumor on an image with an accuracy of 96%. The parameters for this training contain:
 - Batch size of 32
 - 15 epochs of training
 - Learning rate of 0.001

# Transformation into a FL workflow
The code of the use case was transformed into an FL workflow using the provided instructions within the [NVFlare repository](https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/ml-to-fl) 

# Structure of this project

```
├── README.md                   <- The top-level README for developers using this project.
├── code
│   ├── __init__.py             <- Makes <your-model-source> a Python module
│   ├── dataLoader.py           <- Module to load the data and split it for the given amount of sites based on a JSON File
│   ├── mlflow_receiver.py      <- Module for tracking the experiment with MLFlow
│   ├── net.py                  <- The convolutional neural network
│   ├── prepare_data_split.py   <- Module for splitting the data based on the given splitting method
│   └── tumorDetection.py       <- Main file of the federated learning workflow
│
├── job_configs                 <- folder with different FL algorithms to use for this project
│   ├── fedavg                  <- config files and main code for using fedavg as FL algorithm
│   ├── fedopt                  <- config files and main code for using fedopt as FL algorithm
│   ├── fedprox                 <- config files and main code for using fedprox as FL algorithm
│   └── scaffold                <- config files and main code for using scaffold as FL algorithm
├── jobs/decorator              <- folder with configs and code that will be used
└──output                       <- output folder for the JSON file defining the data split
```

# Run the code

To run the code, a virtual environment like [venv](https://docs.python.org/3/library/venv.html) is recommended.
The dataset is not contained within this repository but can be downloaded under https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset
In this code a private MLFlow instance was used. 
To avoid errors the link should be changed to a local MLFlow instance within jobs/decorator/app/config/config_fed_server.conf
When uing a MLFlow server instance, the credentials need to be exported within the terminal with
``` bash
export MLFLOW_TRACKING_USERNAME='username'  
export MLFLOW_TRACKING_PASSWORD='password'  
```
To run the code, start the virtual environment and the NVFlare simulator command

``` bash
source folder/to/venv/bin/activate
nvflare simulator -n 2 -t 2 ./jobs/decorator -w decorator_workspace
```

- `n`: indicates the amount of sites within the project. This number should align with the number of sites within the config files and within the main code
- `t`: indicates the amount of threads

After running the code the folder `decorator_workspace` will be created with different log and config files.
