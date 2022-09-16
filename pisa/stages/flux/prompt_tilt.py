
import numpy as np
import math
import os 
from glob import glob
from pisa.utils.profiler import profile

from numba import njit, prange  # trivially parallelize for-loops

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
from pisa.utils.log import logging

PIVOT = FTYPE(100.0e3)

class prompt_tilt(Stage):
    def __init__(self,
        **std_kwargs):

        expected_params = [ 
            "prompt_deltagamma",
            "prompt_norm"
        ]

        super().__init__(
            expected_params=expected_params,
            **std_kwargs
        )

    def setup_function(self):
        self.data.representation = self.calc_mode

        for container in self.data:
            container["nu_flux"] =  np.full((container.size, 3), np.NaN, dtype=FTYPE)

    @profile
    def compute_function(self):
        for container in self.data:
            nubar = container["nubar"]
            if nubar > 0: flux_key = "nu_flux_nominal"
            elif nubar < 0: flux_key = "nubar_flux_nominal"
            negative_mask = container["nu_flux"] < 0
            if np.any(negative_mask):
                container["nu_flux"][negative_mask] = 0.0


            apply_sys_loop(
                container["true_energy"],
                container[flux_key], 
                self.params.prompt_deltagamma.m_as("dimensionless"),
                self.params.prompt_norm.m_as("dimensionless"),
                out = container["nu_flux"])

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
    nu_flux_nominal,
    delta_index,
    scale,
    out,
):
    """

    """

    n_evts, n_flavs = nu_flux_nominal.shape

    for event in prange(n_evts):
        spec_scale = scale*spectral_index_scale(true_energy[event], PIVOT, delta_index)
        for flav in range(n_flavs):
            out[event, flav] = nu_flux_nominal[event, flav]*spec_scale