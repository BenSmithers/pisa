"""
This stage modifies weights according to the ice gradients it is provided
"""


"""
stage to implement getting the contribution to fluxes from astrophysical neutrino sources
"""
import numpy as np
from math import log10

from pisa.utils.profiler import profile
from pisa import FTYPE, TARGET, PISA_NUM_THREADS
from pisa.utils.resources import find_resource

from pisa.core.stage import Stage

from pisa.utils.log import logging

from numba import njit, prange  # trivially parallelize for-loops

class ice_gradient(Stage):
    """
    Stage to weight the pipeline to an astrophysical flux. 
    This stage stands in place of a typical (flux + osc) stage combo


    Parameters
    ----------
    params
        Expected params are .. ::
            astro_delta : quantity (dimensionless)
            astro_norm : quantity (dimensionless)

    TODO: Add more astrophysical flux implementations
        - Broken power law
        - whatever else is needed... 
    """

    def __init__(self, 
            gradient_0:str,
            gradient_1:str,
            **std_kwargs):
        
        _grad_0_name = find_resource(gradient_0)
        _grad_1_name = find_resource(gradient_1)

        self._grad_0 = np.transpose(np.loadtxt(_grad_0_name))
        self._grad_1 = np.transpose(np.loadtxt(_grad_1_name))

        self._bin_edges = np.array([])
        self._one_sigma_perts_0 = np.array([])
        self._one_sigma_perts_1 = np.array([])

        expected_params = ("ice_grad_0", "ice_grad_1")

        self._correlation = 5.091035738186185100e-02

        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

    def setup_function(self):
        _bin_edges = list(self._grad_0[0]).append(self._grad_0[1][-1])

        grad_0_vals = self._grad_0[-1]
        grad_1_vals = self._grad_1[-1]

        for container in self.data:
            container["grad0_scales"] = grad_0_vals[np.digitize(container["true_energy"], _bin_edges) ]
            container["grad1_scales"] = grad_1_vals[np.digitize(container["true_energy"], _bin_edges) ]

    def apply_function(self):
        for container in self.data:
            container["weights"] *= 1 + self.params.ice_grad_0.value.m_as("dimensionless")*container["grad0_scales"] 
            container["weights"] *= 1 + self.params.ice_grad_1.value.m_as("dimensionless")*container["grad1_scales"] 

            container.mark_changed("weights")