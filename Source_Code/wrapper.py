import numpy as np
import pandas as pd

import time
import os
from sklearn.utils import shuffle
from col_headers import Header
from experiments import Experiments, Configs
from experiments.useful import timestamped_to_vector, unison_shuffled_copies, extract_neg
from tester import *
headers = Header()
c = headers.classification_col
v = headers.vector_col

my_params = { 
    "layer_type": "GRU",         
    "loss": "binary_crossentropy", 
    "kernel_initializer": "lecun_uniform", 
    "recurrent_initializer": "lecun_uniform",
    "activation": "sigmoid",
    "optimiser": "adam",    
    "sequence_length":10,  
    "recurrent_dropout": 0,
    "depth":3,          
    "bidirectional": True, 
    "hidden_neurons":73, 
    "dropout": 0 , 
    "b_l1_reg": 0,          
    "b_l2_reg": 0,          
    "r_l1_reg": 0,         
    "r_l2_reg": 0,         
    "epochs": 70,
    "batch_size": 64
}

os.system('clear')
print('1. Training')
print('2. Prediction')

a = int(input('Enter option:\n'))
while True:
    os.system('CLS')
    if a ==1:
        data = pd.read_csv("testing.csv")
        test = data[data["test_set"] == True]
        train = data[data["test_set"] == False]

        train = train[headers.training_headers].to_numpy()
        test = test[headers.training_headers].to_numpy()
        x_test, y_test = timestamped_to_vector(test, vector_col=v, time_start=0, classification_col=c)
        x_train, y_train = timestamped_to_vector(train, vector_col=v, time_start=0, classification_col=c)
        rand_params = Configs.get_all()

        final(x_train,y_train,x_test,y_test,my_params)
        print('1. Training')
        print('2. Prediction')
        a = int(input('Enter option:\n'))
    elif(a==2):
        path = input('Enter Test Data path : ')

        mal_test = pd.read_csv(path)
        
        mal_test = mal_test[headers.training_headers].to_numpy()
        mal_x,mal_y = timestamped_to_vector(mal_test, vector_col=v, time_start=0, classification_col=c)
        predictor(mal_x,mal_y,my_params)
        time.sleep(5)
        os.system('clear')
        print('1. Training')
        print('2. Prediction')
        a = int(input('Enter option \n'))

    else:
        print('Bye :)')
        time.sleep(5)
        os.system('CLS')
        break

