
from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
import numpy as np
import json
from copy import deepcopy


"""
I chose to also implement correlated parameters in this form - which is a much more computationally efficient implementation

This way we avoid the python call stack as much as possible, and instead now you provide the stage the correlation and the parameters in the already-correlated basis
Then, it does all the work internally 

Users 
"""

class correlated_stage(Stage):
    def __init__(self,
                 correlation_file,
                 as_gradients=False,
                 **std_kwargs):
        if type(self)==correlated_stage:
            raise NotImplemented("This should only be used as a derived class!!")
        
        # this should be a json file where it's a nested dictionary for 
        # param: 
        #   subparm: corrlation
        #   other_subparm : correlation
        _correlation_file =find_resource(correlation_file)
        _obj = open(_correlation_file, 'rt')
        data = json.load(_obj)
        _obj.close()
        self._dim = len(data.keys())
        
        self._correlation = np.zeros(shape=(self._dim, self._dim))
        for i, key in enumerate(data.keys()):
            for j, subkey in enumerate(data[key].keys()):
                self._correlation[i][j] = data[key][subkey]

        self._sigmas, self._inv_t = np.linalg.eig(self._correlation)
        self._sigmas = np.sqrt(self._sigmas)

        self._effects = list(data.keys())
        self._rotated = ["{}_{}".format(self.name_root, i) for i in range(len(self._effects))]
   
        self._param_cache = None
        self._per_event_effects = []
        self._as_gradients = as_gradients # add the effects if True
        # otherwise will do prod(1+eff)
        if self._as_gradients:
            print("Constructing as a gradient effect")
        else:
            print("Constructing as a proportional effect")

        super().__init__(
            expected_params=self._rotated,
            **std_kwargs
        )

    @property
    def name_root(self):
        raise NotImplementedError()

    def compute_function(self):
        """
        Takes the values that are in whole units of sigma
        Scale them up by the eigenvalues of the diagonalized correlation matrix
        Rotate those back into the correlated basis 
        """
        values = np.array([self.params[param].value.m_as("dimensionless") for param in self._rotated])
        self._param_cache = np.matmul(self._sigmas*values, self._inv_t)

    def setup_function(self):
        """
            User must re-implement this function! 

            You need to fill in this "per_event_effects" attribute 
            It should be a 2D numpy array, where the dimensionality is like 
                per_event_effect[ which effect ][ which event ]

            So that per_event_effect[0] would be the 0th effect for all events

            effects are implemented as ratios of the full bin occupation. 
            A value of "0.05" increases the occupation by 5% 
        """
        self._per_event_effects = []
        for container in self.data:
            if container.size==0:
                continue
            per_event_effect = []

            for key in self._effects:
                # build a effect for each of the original (correlated) effects
                per_event_effect.append( np.zeros( container.size ) )
            self._per_event_effects.append( deepcopy(per_event_effect) )

        raise NotImplementedError("Implement this in a derived class")
        
    def apply_function(self):
        count = 0
        for container in self.data:
            if container.size==0:
                continue

            if self._as_gradients:
                effect = self._param_cache*self._per_event_effects[count].T
                container["weights"] = container["weights"] + np.sum(effect, axis=1)

            else:
                effect = 1.0 + self._param_cache*self._per_event_effects[count].T

                container["weights"]=container["weights"]*np.prod(effect, axis=1)
            
            # gradients can make weights below zero
            container["weights"][container["weights"] < 0.0] = 0.0
            count += 1