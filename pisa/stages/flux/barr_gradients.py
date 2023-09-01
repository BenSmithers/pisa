"""
Another Barr stage

This one loads in various nusquids gradients with which it 
"""

import numpy as np
import math
import os 
from glob import glob
from pisa.utils.profiler import profile

from scipy.interpolate import RectBivariateSpline

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
from pisa.utils.log import logging

import h5py as h5

PIVOT = FTYPE(2020)


class barr_gradients(Stage):
    def __init__(
        self,
        grad_folder:str,
        file_template:str,
        **std_kwargs):

        self._gradient_folder = find_resource(grad_folder)
        self.file_template = file_template
        self._barr_params=[
            "WP",
            "WM",
            "YP",
            "YM",
            "ZP",
            "ZM"
        ]

        expected_params = self._barr_params + ["deltagamma", "conv_norm"]


        super().__init__(
            expected_params=expected_params,
            **std_kwargs
        )

    def setup_function(self):
        
        self.data.representation = self.calc_mode

        logging.debug("Setting up barr gradient weighter stage")
        grads = {}

        for param in self._barr_params:
            filename = os.path.join(
                self._gradient_folder,
                self.file_template.format(param)
            ) 

            dfile = h5.File(filename, 'r')
            grads[param]= {}
            for key in dfile.keys():
                grads[param][key] = np.array(dfile[key][:])
            print("Loaded {}".format(filename))

        grad_sum = [0.0 for i in range(len(self._barr_params))]
        total_evt = [0 for i in range(len(self._barr_params))]

        self._per_event_effects = []
        for container in self.data:
            if container.size==0:
                continue

            container["power_scale"] = np.zeros(container.size)
            per_event_effect = []
            for ip, param in enumerate(self._barr_params):
                
                """
                    evaluate the gradient for each of these 
                """
                key = ""
                if container["nubar"]<0:
                    key+="antinu"
                else:
                    key+="nu"
                if container["flav"]==0:
                    key+="e"
                elif container["flav"]==1:
                    key+="mu"
                else:
                    key+="tau"
                
                interp = RectBivariateSpline( grads[param]["costh_nodes"], np.log10(grads[param]["energy_nodes"]),grads[param]["conv_"+key])
                entry = interp(container["true_coszen"], np.log10(container["true_energy"]), grid=False)
                per_event_effect.append(entry)
      
            self._per_event_effects.append(np.array(per_event_effect))
        

    def compute_function(self):
        self._param_cache = np.array([self.params[param].value.m_as("dimensionless") for param in self._barr_params])
        for container in self.data:
            if container.size==0:
                continue
            container["power_scale"] = self.params.conv_norm.value.m_as("dimensionless")*((container["true_energy"]/PIVOT)**(self.params.deltagamma.value.m_as("dimensionless")))


    def apply_function(self):
        count = 0
        for container in self.data:
            if container.size==0:
                continue

            effect = self._param_cache*self._per_event_effects[count].T

            # a little hacky, but we do this to get a shape-only effect 
            norm = np.sum(container["weights"]*container['weighted_aeff'])
            container["weights"] = container["weights"] + np.sum(effect, axis=1)
            post_norm = np.sum(container["weights"]*container['weighted_aeff'])
            container["weights"] *= norm/post_norm

            container["weights"][container["weights"]<0] = 0.0
            container["weights"] *= container["power_scale"] # force a shape-only effect for the gradients 
            count += 1
            
                                    