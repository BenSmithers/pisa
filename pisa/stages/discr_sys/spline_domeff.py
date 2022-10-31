"""
PISA stage using pre-calculated domeff splines to modify neutrino fluxes 
"""
import numpy as np
import math
import os 
from glob import glob


from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.profiler import profile
from pisa.utils.resources import find_resource
from pisa.utils.log import logging


import photospline as ps

class spline_domeff(Stage):
    """
    This stage uses pre-calculated domefficiency splines to reweight your MC to different effective dom efficiencies 
    """
    def __init__(
        self,
        spline_folder:str,
        matching:str,
        central_domeff=1.27,
        **std_kwargs):

        self._spline_folder = find_resource(spline_folder)
        self._matching = matching
        self._central_domeff = central_domeff

        expected_params=[ 
            "domeff"
        ]

        super().__init__(
            expected_params=expected_params,
            **std_kwargs
        )


        self._spline_names = glob(os.path.join(self._spline_folder, "*.fits"))
        self._spline_names = list(filter(lambda key:self._matching in key, self._spline_names))
        if not (len(self._spline_names)>=1):
            raise ValueError("Could not find any files matching '{}'".format(self._matching))


        self._splinetable = {}

    def setup_function(self):
        """
        Load in the DOMEff splines, fill in the cache 
        """
        logging.debug("setting up spline_domeeff stage")
        # this should link all the containers, if it doesn't, make a scene 

        for use in ["track", "shower"]:
            spline_subset = list(filter(lambda key: use in key, self._spline_names))
            if not len(spline_subset)==1:
                logging.debug(spline_subset)
                logging.warn("Could not find any splines for {}".format(use))
                continue

            self._splinetable[use] = ps.SplineTable(spline_subset[0])


        for container in self.data:
            container["domeff_splines"] = np.ones(container.size)

            if container.size==0:
                continue
            
            if container.name in ["numu_cc", "numubar_cc"]:
                use = "track"
            else:
                use = "shower"
            if use not in self._splinetable:
                logging.fatal("Could not find appropriate splines to weight {}".format(container.name))

            container["domeff_cache"] = self._splinetable[use].evaluate_simple((
                np.log10(container["true_energy"]),
                container["true_coszen"],
                [self._central_domeff]
                ))

    @profile
    def compute_function(self):


        for container in self.data:
            if container.name in ["numu_cc", "numubar_cc"]:
                use = "track"
            else:
                use = "shower"

            if container.size==0:
                continue
            logging.debug("about to get the rate {}".format(self.params.domeff.value))
        
            rate = self._splinetable[use].evaluate_simple((
                np.log10(container["true_energy"]),
                container["true_coszen"],
                [self.params.domeff.value.m_as("dimensionless")]
            ))
            logging.debug("pow time")
            scales = np.power(10.0, rate - container["domeff_cache"] )
            mask = scales < 0.0
            scales[mask] = 0.0

            container["domeff_splines"] = scales
            container.mark_changed("domeff_splines")

            logging.debug("reweight pisa")
    
    def apply_function(self):
        for container in self.data:
            container["weights"] = container["weights"]*container["domeff_splines"]

            container.mark_changed("weights")