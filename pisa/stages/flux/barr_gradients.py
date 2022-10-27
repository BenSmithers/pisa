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

PIVOT = FTYPE(100.0e3)


class barr_gradients(Stage):
    def __init__(
        self,
        grad_folder:str,
        **std_kwargs):

        self._gradient_folder = find_resource(grad_folder)
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
            dfile = h5.File(os.path.join(self._gradient_folder, param+_suffix),'r')
            grads[param]= {}
            for key in dfile.keys():
                grads[param][key] = np.array(dfile[key][:])

        for param in self._barr_params:
            for container in self.data:
                """
                    evaluate the gradient thingy for each of these 
                """
                container["barr_scale"] = np.zeros(container.size)
                container[param] = np.zeros(container.size)
                                
                if container.size==0:
                    continue
                
                if container["nubar"]<0:
                    e_effect = RectBivariateSpline( grads[param]["costh_nodes"], grads[param]["energy_nodes"],grads[param]["nuebar"])
                    mu_effect = RectBivariateSpline( grads[param]["costh_nodes"], grads[param]["energy_nodes"],grads[param]["numubar"])
                else:
                    e_effect = RectBivariateSpline( grads[param]["costh_nodes"], grads[param]["energy_nodes"],grads[param]["nue"])
                    mu_effect = RectBivariateSpline( grads[param]["costh_nodes"], grads[param]["energy_nodes"],grads[param]["numu"])
                
                #logging.debug("shape out is {} versus {} expected".format(np.shape(out), container.size))
                # no tau effect since there are no conventional taus 
                container[param] = (e_effect(container["true_coszen"], container["true_energy"], grid=False)*container["flux_contribution"][:,0] 
                                        + mu_effect(container["true_coszen"], container["true_energy"], grid=False)*container["flux_contribution"][:,1])

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
            
            container.mark_changed("barr_scale")

    def apply_function(self):
        for container in self.data:
            if container.size==0:
                continue
            container["weights"] = self.params.conv_norm.value.m_as("dimensionless")*(container["weights"] + container["barr_scale"]) \
                                    *np.power(container["true_energy"]/PIVOT, self.params.deltagamma.value.m_as("dimensionless"))
            container.mark_changed("weights")