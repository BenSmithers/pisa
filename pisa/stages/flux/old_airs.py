"""
This implemenents the splined hole ice that the original MEOWS analysis used
"""



from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE, myjit
from pisa.utils.resources import find_resource
from math import pi
import photospline

import numpy as np


class old_airs(Stage):
    """
    Parameters
    ----------
    airs_spline : spline containing the 1-sigma shifts from AIRS data

    params : ParamSet
        Must exclusively have parameters: .. ::

            scale : quantity (dimensionless)
                the scale by which the weights are perturbed via the airs 1-sigma shift
    """

    def __init__(self, 
            airs_spline, 
            **std_kwargs):

        self.airs_spline = find_resource(airs_spline)        

        expected_params = [
            "airs_scale",
        ]

        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

    def setup_function(self):
        """
        Uses the splines to quickly evaluate the 1-sigma perturbtations at each of the events
        """

        self.spline_table = photospline.SplineTable(self.airs_spline)

        # consider 'reco_coszen" and 'reco_energy' containers
        for container in self.data:
            if container.size==0:
                continue
            re = 6.37814e6
            depth = 2500

            theta = container["true_coszen"]

            # get the mceq angle, then convert that back to the normal zenith 

            eff_zen = theta*1.0
            eff_zen[eff_zen>0.1] = 0.1
            container["airs_cache"] = self.spline_table.evaluate_simple(
                (np.log10(container["true_energy"]), theta) ## mirror it around coszen=0
            )            
    
    def apply_function(self):
        for container in self.data:
            if container.size==0:
                continue
            container["weights"] = container["weights"]*(1 + container["airs_cache"]*self.params.airs_scale.value.m_as("dimensionless"))