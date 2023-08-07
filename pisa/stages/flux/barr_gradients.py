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

from numba import njit, prange  # trivially parallelize for-loops



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
        _suffix = ".hdf5"
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

        for container in self.data:
            if container.size==0:
                continue

            container["power_scale"] = np.zeros(container.size)
            
            for param in self._barr_params:
                
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
                container[param] = interp(container["true_coszen"], np.log10(container["true_energy"]), grid=False)

    @profile
    def compute_function(self):
        for container in self.data:
            if container.size==0:
                continue
            # modify container[flux_key]
            container["barr_scale"] = ( 
                container["WP"]*self.params.WP.value.m_as("dimensionless") + 
                container["WM"]*self.params.WM.value.m_as("dimensionless") + 
                container["YP"]*self.params.YP.value.m_as("dimensionless") +
                container["YM"]*self.params.YM.value.m_as("dimensionless") +
                container["ZP"]*self.params.ZP.value.m_as("dimensionless") +
                container["ZM"]*self.params.ZM.value.m_as("dimensionless") 
                ) 
            container["power_scale"] = ((container["true_energy"]/PIVOT)**(self.params.deltagamma.value.m_as("dimensionless")))*self.params.conv_norm.value.m_as("dimensionless")


    def apply_function(self):
        for container in self.data:
            if container.size==0:
                continue
            container["weights"] += container["barr_scale"]
            container["weights"] *= container["power_scale"]
            container["weights"][container["weights"]<0] = 0.0
                                    