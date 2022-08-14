

from cgi import test
import numpy as np
import random
from sklearn.metrics import mean_squared_error

from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from experiments.RNN import generate_model
from experiments.useful import *
random.seed(12)
np.random.seed(12)


def final(x_train,y_train,x_test,y_test,params):
    #x_train, y_train = truncate_and_tensor(x[train_idxs], y[train_idxs], current_params["sequence_length"])
    x_train, y_train = remove_short(x_train, y_train, params["sequence_length"])
    x_test, y_test = remove_short(x_test, y_test, params["sequence_length"])
    
    x_train, y_train = unison_shuffled_copies([x_train, y_train])
    x_test, y_test = unison_shuffled_copies([x_test, y_test])  
    
    means, stdvs = get_mean_and_stdv(x_train)
    
    x_train = scale_array(x_train, means, stdvs)
    x_test = scale_array(x_test, means, stdvs)
    model = generate_model(x_train,y_train,params)
    reset_states = ResetStatesCallback()
    model.fit(x_train, y_train, batch_size=params["batch_size"], epochs=params["epochs"],shuffle=True,	verbose=0,callbacks=[reset_states])    
    model.save('rnn.h5')
    del model
    model = load_model('rnn.h5')
    pred_Y = model.predict(x_test, batch_size=params["batch_size"])
    print(np.round(pred_Y))
    print("")
    print('-------------------- Model Summary --------------------')
    model.summary() # print model summary   
    print('---------- Evaluation on Test Data ----------')
    print("MSE: ", mean_squared_error(y_test, np.round(pred_Y)))
    print("")
    print(accuracy_score(y_test,np.round(pred_Y)))    
    np.save('means.npy',means)
    np.save('std.npy',stdvs)

def predictor(a,b,params):
    #print(a)
    means = np.load('means.npy')
    stdvs = np.load('std.npy')
    a,b = remove_short(a,b, length=params["sequence_length"])
    a = scale_array(a, means, stdvs)
    model = load_model('rnn.h5')
    #print(a)
    pred_Y = model.predict(a, batch_size=params["batch_size"])

    for i in pred_Y:
        if i[0]>=0.5:
            print('Malware')
        else:
            print('Benign')
