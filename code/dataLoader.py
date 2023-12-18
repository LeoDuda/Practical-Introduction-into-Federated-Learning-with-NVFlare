import os
import json
import numpy as np
import pandas as pd
from PIL import Image


class dataLoader():

    #for testing purposes
    json_dir = os.path.realpath('json')  # path of image directory
    with open(json_dir + '/data_site1.json', 'r') as f:
        data = json.load(f)

    print(data)
    
    
    
    
    #load the json file and the client id
    def load_data(json: dict, client_id: str):

        data_index = json ["data_index"]
        img_dir = os.path.realpath('dataset')  # path of image directory
        images = os.listdir(img_dir)

        if client_id not in data_index.keys() :
            raise ValueError(
                f"{client_id} not found",
            )
        
        if "valid" not in data_index.keys() :
            raise ValueError(
                "no validation split found",
            )


        data = [Image.open(os.path.join(img_dir, i)).convert('L').resize((256, 256)) for i in images]
        data = np.array([np.array(d).reshape(-1) for d in data])

        
        tumor_df = None
        img_names, img_labels = zip(*[(i, 0 if 'Not Cancer' in i else 1) for i in images[data_index[client_id]["start"]:data_index[client_id]["end"]]])
        names = pd.Series(img_names, name='name')
        labels = pd.Series(img_labels, name='label')
        tumor_df = pd.concat([names,labels],axis=1)

        #for testing purposes
        print(images[data_index[client_id]["start"]:data_index[client_id]["end"]])
        print(len(tumor_df))
        print(data_index[client_id]["end"] - data_index[client_id]["start"])




        
    load_data(data, "site1")
    load_data(data, "site2")