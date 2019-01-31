import numpy as np
import pdb

class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
    
        
        '''
        N = cmap.shape[0]
        M = cmap.shape[1]
        
        #Building Transition Matrix based on Probability 0.9 if true and 0.1 if false
        right = 0.9*np.eye(M-1) 
        right_new = np.append(np.zeros([M-1,1]),right,axis=1)
        right_new = np.append(right_new,np.zeros([1,M]),axis=0)
        markov = right_new+0.1*np.eye(M)
        markov[-1,-1]=1
        
        #ACTION UPDATE
        if action[0]==1:
            #going right
            new_bel = np.matmul(belief,markov)
        elif(action[0]==-1):
            #going left
            new_bel = np.flip(belief,1)
            new_bel = np.flip(np.matmul(new_bel,markov),1)
        elif(action[1]==1):
            #going up
            new_bel = np.flip(belief.T,1)
            new_bel = np.flip(np.matmul(new_bel,markov),1).T
        else:
            #going down
            new_bel = belief.T
            new_bel = np.matmul(new_bel,markov).T
#        pdb.set_trace()
        #MEASUREMENT UPDATE
        #Update bel based on action
        if (observation ==1):
            
            ind = np.where(cmap==0)
            a = 0.9*np.ones([N,M])
            a[ind]=0.1
            
        elif(observation ==0):
            ind = np.where(cmap==0)
            a = 0.1*np.ones([N,M])
            a[ind]=0.9
        
        #Update bel based on measurement
        new_bel = a*new_bel
        
        ind = np.unravel_index(np.argmax(new_bel, axis=None), new_bel.shape)
        

        return new_bel/np.sum(new_bel), [ind[1],M-ind[0]-1]
        