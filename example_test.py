import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random
import pdb

if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    bel_init = 1/400 * np.ones([20,20])
    
    obj1 = HistogramFilter()
    for i in range(0,30):
        u = actions[i]
        z = observations[i]
        bel_init,ind = obj1.histogram_filter(cmap, bel_init, u , z)
        
        
        
        
        