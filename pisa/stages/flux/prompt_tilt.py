
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

PIVOT = FTYPE(2020)

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
            container["nu_flux"] =  np.ones(container.size, dtype=FTYPE)



    @profile
    def apply_function(self):
        for container in self.data:
            container["weights"] = (container["weights"]
                *np.power(
                    container["true_energy"]/PIVOT, 
                    self.params.prompt_deltagamma.m_as("dimensionless")
                )
                *self.params.prompt_norm.m_as("dimensionless")
            )