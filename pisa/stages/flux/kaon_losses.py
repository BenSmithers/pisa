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

        self._kaon_root_name = "kaon_losses_1s_"
        self._kaon_spline_folder = find_resource(kaon_spline)

        expected_params = [
            "kaon_scale",
        ]

        super().__init__(
            expected_params=expected_params,
            **std_kwargs
        )

    def assemble_name(self, flavor, neutrino)->str:
        suffix = ""
        if neutrino==-1:
            suffix = "antinu"
        elif neutrino==1:
            suffix = "nu"
        else:
            logging.fatal("What kind of neutrino is {}?".format(neutrino))
        
        if flavor==0:
            suffix+="e"
        elif flavor==1:
            suffix+="mu"
        elif flavor==2:
            suffix+="tau"
        else:
            logging.fatal("Unknown flavor {}".format(flavor))

        full_name = os.path.join(
            self._kaon_spline_folder,
            self._kaon_root_name + suffix + ".fits"
        )

        return full_name

    def setup_function(self):
        """
        Pre-compute the 1-sigma shifts 
        """
        for container in self.data:
            if container["flav"]==2: # no tau!
                container["kaon_1s_perturb"] = np.zeros(container.size, dtype=FTYPE)
                continue

            kaon_spline = photospline.SplineTable(self.assemble_name(container["flav"], container["nubar"]))

            container["kaon_1s_perturb"] = np.zeros(container.size, dtype=FTYPE)

            if container.size!=0:
                container["kaon_1s_perturb"] = kaon_spline.evaluate_simple(
                    (np.log10(container["true_energy"]), container["true_coszen"])
                )

            container.mark_changed("kaon_1s_perturb")
    
    def apply_function(self):
        for container in self.data:
            container["weights"] *= (1.0 + 
                container["kaon_1s_perturb"] * self.params.kaon_scale.value.m_as("dimensionless"))

            container.mark_changed("weights")