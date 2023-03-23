"""
Stage to implement the atmospheric density uncertainty. 

Ben Smithers
"""

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.resources import find_resource

import photospline

import numpy as np
import h5py as h5
import os 
from scipy.interpolate import RectBivariateSpline



class airs(Stage):
    """
    Note that this stage is just about identical to the kaon loss stage.

    Parameters
    ----------
    airs_spline : hdf5 file containing the 1-sigma shifts from AIRS data

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
        airs_file = h5.File(self.airs_spline, 'r')

        for container in self.data:
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
            
            interpolator = RectBivariateSpline( airs_file["costh_nodes"], np.log10(airs_file["energy_nodes"]), airs_file["conv_"+key]) 

            container["airs_1s_perturb"] = np.zeros(container.size, dtype=FTYPE)

            if container.size!=0:
                container["airs_1s_perturb"] = interpolator(
                    container["true_coszen"],
                    np.log10(container["true_energy"]),
                    grid=False)


            container.mark_changed("airs_1s_perturb")
        

    @profile
    def apply_function(self):
        """
        Modify the weights according to the new scale parameter!
        """
        for container in self.data:
            container["weights"] += container["airs_1s_perturb"] * self.params.airs_scale.value.m_as("dimensionless")
            container["weights"][container["weights"]<0] = 0.0

            container.mark_changed("weights")