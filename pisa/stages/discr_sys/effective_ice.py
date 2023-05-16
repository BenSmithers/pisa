"""
Implements a stage to bring in a json file for the effective 2D ice gradients! 

This guy is a little tricky
We have three (or more, or fewer) effective 2D gradients which can be used, with a correlation matrix, to represent the same rank-4 analysis covariance tensor that is normally constructed with the nine (or more) original amplitude and phase gradient surfaces

So here we load those effective ones in, but then we construct a new basis set of parameters tha tweak the effective gradients.
This keeps any extra calls to the python stack to a minimum, hurray!  
"""

from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
from copy import deepcopy
import numpy as np
import json 


class effective_ice(Stage):
    """
    Stage to implement what is written above

    Takes only a path to a json file. That configures this with thich parameters it will expect 

    Users will provide just the parameters as if they range from [-5,+5] sigmas, so nothing fancy
    These will represent the effective, effective (yes, 2x) gradients. 
    Internally those parameters are scaled by the sigmas in the uncorrelated basis 
    And then we rotate into 
    """

    def __init__(self, 
            gradient_file:str,
            **std_kwargs):

        _gradient_file = find_resource(gradient_file)

        _obj = open(_gradient_file,'rt')
        data = json.load(_obj)
        _obj.close()

        self._e_bins = np.power(10, data["e_bins"])
        self._c_bins = np.array(data["c_bins"])

        self._correlation = np.array(data["cor"])

        self._sigmas, self._inv_t = np.linalg.eig(self._correlation)

        # get the length of all non-bin based keys
        self._grad_keys = list(filter(lambda key: "bins" not in key and "cor" not in key, data.keys()))
        
        self._gradients = {}
        for key in  self._grad_keys:
            self._gradients[key] = np.array(data[key]).flatten()

        super().__init__(
            expected_params=self._grad_keys,
            **std_kwargs
        )

        self._param_cache = None

    def compute_function(self):
        """
            Takes the values that are in whole units of sigma
            Scale them up by the eigenvalues of the diagonalized correlation matrix
            Rotate those back into the correlated basis
            """
        values = np.array([self.params[param].value.m_as("dimensionless") for param in self._grad_keys])
        self._param_cache = np.matmul(self._sigmas*values, self._inv_t)

    def setup_function(self):
 
        
        self._per_evt_ice_grads = []
        
        for container in self.data:
            if container.size==0:
                continue
            energy_indices = np.digitize(np.log10(container["reco_energy"]), self._e_bins)
            energy_indices[energy_indices==0] = 1
            energy_indices[energy_indices==len(self._e_bins)] = len(self._e_bins) -1
            energy_indices-=1

            zenith_indices =np.digitize(container["reco_coszen"], self._c_bins)
            zenith_indices[zenith_indices==0] = 1
            zenith_indices[zenith_indices==len(self._c_bins)] = len(self._c_bins) -1
            zenith_indices-=1

            net_indices = zenith_indices+ energy_indices*(len(self._c_bins)-1)
            

            per_evt_grad =[]
            for key in self._grad_keys:
                per_evt_grad.append(self._gradients[key][net_indices])
            self._per_evt_ice_grads.append(deepcopy(np.array(per_evt_grad)))
    

    def apply_function(self):
        ic = 0
        for container in self.data:
            if container.size==0:
                continue
        
            effect = 1.0 + self._param_cache*self._per_evt_ice_grads[ic].T
            #print(np.shape(effect))

   
            container["weights"]=container["weights"]*np.prod(effect, axis=1)
            ic+=1

