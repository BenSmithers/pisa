"""
Stage to implement the effects of uncertainty on kaon-nucleon interaction cross section
This effects the flux, and therefore the weights of events.
Loss gradients must be calculated ahead of time and stored in splines!

It's the kaon losses stage! 

Ben Smithers
"""
import profile
import photospline 

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.resources import find_resource

import numpy as np
import os
import h5py as h5
from scipy.interpolate import RectBivariateSpline

"""
TODO: make a generic "spline weight stage" that generalizes this and the AIRS spline stage 
"""

class kaon_losses(Stage):
    """
    The ~kaon energy loss~ systematic!
    Should only be used in the conventional neutrino pipelines 

    Parameters
    ----------

    kaon_spline : spline containing the 1-sigma shifts from kaon losses

    params: ParamSet
        Must have parameters: .. ::
            scale : quantity (dimensionless)
                the scale by wich the weights are perturbed (1.0 is a 1sigma perturbation on kaon-nucleon xs)


    correction is calculated using 1.0 + spline_eval*scale 
    """

    def __init__(self,
        kaon_spline,
        **std_kwargs):

        #self.kaon_spline = find_resource(kaon_spline)
        self.kaon_spline = kaon_spline

        expected_params = [
            "kaon_scale",
        ]

        super().__init__(
            expected_params=expected_params,
            **std_kwargs
        )

    def setup_function(self):
        """
        Pre-compute the 1-sigma shifts 
        """

        kaon_file = h5.File(self.kaon_spline, 'r')

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
            
            interpolator = RectBivariateSpline( kaon_file["costh_nodes"], np.log10(kaon_file["energy_nodes"]), kaon_file["conv_"+key]) 

            container["kaon_1s_perturb"] = np.zeros(container.size, dtype=FTYPE)

            if container.size!=0:
                container["kaon_1s_perturb"] = interpolator(
                    container["true_coszen"],
                    np.log10(container["true_energy"]),
                    grid=False)

            container.mark_changed("kaon_1s_perturb")
    
    def apply_function(self):
        for container in self.data:
            container["weights"] += container["kaon_1s_perturb"] * self.params.kaon_scale.value.m_as("dimensionless")

            container.mark_changed("weights")