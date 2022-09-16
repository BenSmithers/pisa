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

        expected_params = self._barr_params + ["deltagamma",]


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

                container[param] = np.zeros(container.size)
                
                container["nu_flux"] = np.full((container.size, 3), np.NaN, dtype=FTYPE)



                if container.size==0:
                    continue
                
                interp = RectBivariateSpline( grads[param]["costh_nodes"], grads[param]["energy_nodes"],grads[param][container.name.split("_")[0]])
                
                #logging.debug("shape out is {} versus {} expected".format(np.shape(out), container.size))
                container[param] =  interp(container["true_coszen"], container["true_energy"], grid=False)

                container.mark_changed(param)

    @profile
    def compute_function(self):
        

        for container in self.data:
            nubar = container["nubar"]
            if nubar > 0: flux_key = "nu_flux_nominal"
            elif nubar < 0: flux_key = "nubar_flux_nominal"
            negative_mask = container["nu_flux"] < 0
            if np.any(negative_mask):
                container["nu_flux"][negative_mask] = 0.0

            # modify container[flux_key]

            scale = (
                container["barr_grad_WP"]*self.params.barr_grad_WP.value.m_as("dimensionless") + 
                container["barr_grad_WM"]*self.params.barr_grad_WM.value.m_as("dimensionless") +   
                container["barr_grad_YP"]*self.params.barr_grad_YP.value.m_as("dimensionless") + 
                container["barr_grad_YM"]*self.params.barr_grad_YM.value.m_as("dimensionless") + 
                container["barr_grad_ZP"]*self.params.barr_grad_ZP.value.m_as("dimensionless") + 
                container["barr_grad_ZM"]*self.params.barr_grad_ZM.value.m_as("dimensionless") 
                )

            apply_sys_loop(
                container["true_energy"],
                container["true_coszen"],
                self.params.deltagamma.value.m_as("dimensionless"),
                PIVOT,
                container[flux_key],
                scale,
                out=container["nu_flux"],
            )

            container.mark_changed("nu_flux")
            negative_mask = container["nu_flux"] < 0
            if np.any(negative_mask):
                container["nu_flux"][negative_mask] = 0.0

            container.mark_changed("nu_flux")


@njit
def spectral_index_scale(true_energy, energy_pivot, delta_index):
    """
      Calculate spectral index scale.
      Adjusts the weights for events in an energy dependent way according to a
      shift in spectral index, applied about a user-defined energy pivot.
      """
    return np.power((true_energy / energy_pivot), delta_index)


@njit(parallel=True if TARGET == "parallel" else False)
def apply_sys_loop(
    true_energy,
    true_coszen,
    delta_index,
    energy_pivot,
    nu_flux_nominal,
    gradients,
    out,
):
    """
    Calculation:
      1) Start from nominal flux
      2) Apply spectral index shift
      3) Add contributions from MCEq-computed gradients

    Array dimensions :
        true_energy : [A]
        true_coszen : [A]
        nubar : scalar integer
        delta_index : scalar float
        energy_pivot : scalar float
        nu_flux_nominal : [A,B]
        gradients : [A,B,C]
        gradient_params : [C]
        out : [A,B] (sys flux)
    where:
        A = num events
        B = num flavors in flux (=3, e.g. e, mu, tau)
        C = num gradients
    Not that first dimension (of length A) is vectorized out
    """

    n_evts, n_flavs = nu_flux_nominal.shape

    for event in prange(n_evts):
        spec_scale = spectral_index_scale(true_energy[event], energy_pivot, delta_index)
        for flav in range(n_flavs):
            out[event, flav] = nu_flux_nominal[event, flav] * spec_scale
            out[event, flav] += gradients[event]
