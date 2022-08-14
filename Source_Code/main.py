import numpy as np
import pandas as pd
from copy import deepcopy
import csv
import gc
import random
from sklearn.utils import shuffle
from col_headers import Header
from experiments import Experiments, Configs
from experiments.useful import timestamped_to_vector, unison_shuffled_copies, extract_neg
from tester import *
# Load data
headers = Header()
c = headers.classification_col
v = headers.vector_col
data = pd.read_csv("testing.csv")
#mal_test = pd.read_csv('test.csv')

test = data[data["test_set"] == True]
train = data[data["test_set"] == False]

train = train[headers.training_headers].to_numpy()
test = test[headers.training_headers].to_numpy()
x_test, y_test = timestamped_to_vector(test, vector_col=v, time_start=0, classification_col=c)
x_train, y_train = timestamped_to_vector(train, vector_col=v, time_start=0, classification_col=c)
#means, stdvs = get_mean_and_stdv(x_train)
rand_params = Configs.get_all()


expt = Experiments.Experiment(rand_params, search_algorithm="random",
   data=(x_train, y_train), folds=10, folder_name="random_search_reults", 
   thresholding=True, threshold=0.5)
expt.run_experiments(500)

