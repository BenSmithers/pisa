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
        **std_kwargs):

        self._gradient_folder = find_resource(grad_folder)
        self.model_label = "0_0.000000_0.000000_0.000000_0.000000_0.000000_0.000000"
        self._barr_params=[
            "barr_grad_WP",
            "barr_grad_WM",
            "barr_grad_YP",
            "barr_grad_YM",
            "barr_grad_ZP",
            "barr_grad_ZM"
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
            dfile = h5.File(os.path.join(self._gradient_folder,param.split("_")[-1], param+"_"+self.model_label+_suffix),'r')
            grads[param]= {}
            for key in dfile.keys():
                grads[param][key] = np.array(dfile[key][:])

        for param in self._barr_params:
            for container in self.data:
                """
                    evaluate the gradient thingy for each of these 
                """
                container["barr_scale"] = np.zeros(container.size)
                container["power_scale"] = np.zeros(container.size)
                container[param] = np.zeros(container.size)

                if container.size==0:
                    continue

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

                #logging.debug("shape out is {} versus {} expected".format(np.shape(out), container.size))
                container[param] = interp(container["true_coszen"], np.log10(container["true_energy"]), grid=False)

                container.mark_changed(param)

    @profile
    def compute_function(self):
        for container in self.data:
            # modify container[flux_key]
            container["barr_scale"] = ( 
                container["barr_grad_WP"]*self.params.barr_grad_WP.value.m_as("dimensionless") + 
                container["barr_grad_WM"]*self.params.barr_grad_WM.value.m_as("dimensionless") + 
                container["barr_grad_YP"]*self.params.barr_grad_YP.value.m_as("dimensionless") +
                container["barr_grad_YM"]*self.params.barr_grad_YM.value.m_as("dimensionless") +
                container["barr_grad_ZP"]*self.params.barr_grad_ZP.value.m_as("dimensionless") +
                container["barr_grad_ZM"]*self.params.barr_grad_ZM.value.m_as("dimensionless") 
                ) 
            #container["barr_scale"][container["barr_scale"] < 0.0]=0.0
            container["power_scale"] = np.power(container["true_energy"]/PIVOT, -1*self.params.deltagamma.value.m_as("dimensionless"))
            container.mark_changed("barr_scale")
            container.mark_changed("power_scale")

    def apply_function(self):
        for container in self.data:
            if container.size==0:
                continue
            container["weights"] = self.params.conv_norm.value.m_as("dimensionless")*(container["weights"] + container["barr_scale"])*container["power_scale"]
                                    
            container.mark_changed("weights")