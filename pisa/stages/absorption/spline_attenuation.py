"""
PISA stage using pre-calculated attenuation splines to modify neutrino fluxes
"""
import numpy as np
import math
import os 
from glob import glob


from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
from pisa.utils.log import logging


import photospline as ps

class spline_attenuation(Stage):
    """
    Modifies nu/nubar fluxes with a separate term
    Uses unique spline for each different flavor/neutrino type 
    Must be provided a key 'matching' against which it will compare against the files it finds in the spline_folder 
    """
    def __init__(
        self, 
        spline_folder:str,
        matching:str,
        **std_kwargs):

        self._spline_folder = find_resource(spline_folder)
        self._matching = matching

        expected_params=[
            "nu_scale",
            "nubar_scale"
        ]

        super().__init__(
            expected_params=expected_params,
            **std_kwargs
        )
        self._spline_names = glob(os.path.join(self._spline_folder, "*.fits"))
        self._spline_names = list(filter(lambda key:self._matching.lower() in key.lower(), self._spline_names))
        logging.debug("all names: {}".format(self._spline_names))
        if not (len(self._spline_names)>=1):
            raise ValueError("Could not find any files matching '{}'".format(self._matching))

    def setup_function(self):
        """
        Load in all the splines
        """

        logging.debug("setting up spline attenuation stage")
        self._splines = {}
            

        for container in self.data:
            """
            loops over non-linked containers and the ones that things made linked 
            """     
            name =container.name.split("_")[0]
            logging.debug("Checking {}".format(name))
            subset = list(filter(lambda key: name in key, self._spline_names))

            container["spline_scales"] = np.ones(container.shape, dtype=FTYPE)

            assert len(subset)>=1, "found odd number of matches: {}".format(subset)
            if len(subset)==1:
                self._splines[name] = ps.SplineTable(subset[0])
                continue

            logging.debug("choosing between {}".format(subset))
            # otherwise we're in a nue/num/nutau like container 
            subset = list(filter(lambda key: "bar" not in key, subset))
            assert len(subset)==1, "Should only be one match: found {}".format(subset)
            self._splines[name]=ps.SplineTable(subset[0])

            

    def compute_function(self):
        
        for container in self.data:
            name =container.name.split("_")[0]
            if container.size==0:
                continue
            factor = (int(container["nubar"]<0)*self.params.nubar_scale.value.m_as("dimensionless")
                        + int(container["nubar"]>0)*self.params.nu_scale.value.m_as("dimensionless"))

            container["spline_scales"] = self._splines[name].evaluate_simple((
                    np.log10(container["true_energy"]),
                    container["true_coszen"],
                    [factor,]
            ))
            container.mark_changed("spline_scales")

    def apply_function(self):
        for container in self.data:
            container["weights"] = container["weights"]*container["spline_scales"]

            container.mark_changed("weights")
