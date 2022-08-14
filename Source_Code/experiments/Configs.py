import numpy as np
from .useful import merge_two_dicts


fixed_parameters = { 
    "layer_type": ["GRU"],         
    "loss": ["binary_crossentropy"], 
    "kernel_initializer": ["lecun_uniform"], 
    "recurrent_initializer": ["lecun_uniform"],
    "activation": ["sigmoid"],
    "optimiser": ["adam"],    
     
    "recurrent_dropout": [0]
}

def get_all():
    all_options =  {
        "depth": [1,2,3],          
        "bidirectional": [True, False], 
        "hidden_neurons": list(range(10, 100,10)), 
        "dropout": [0,0.1,0.2,0.3] , 
        "b_l1_reg": [0, 0.01],       
        "b_l2_reg": [0, 0.01],          
        "r_l1_reg": [0, 0.01],
        "sequence_length":[7,10,9,8,11,15,18],          
        "r_l2_reg": [0, 0.01],         
        "epochs": list(range(40, 100,20)), 
        "batch_size": [32*(2**i) for i in range(3)] 
    }
    return merge_two_dicts(all_options, fixed_parameters)
        