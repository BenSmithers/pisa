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
        logging.debug("setting up spline_domeff stage")
        # this should link all the containers, if it doesn't, make a scene 

        for use in ["track", "shower"]:
            spline_subset = list(filter(lambda key: use in key, self._spline_names))
            if len(spline_subset)==0:
                logging.warn("Did not find any splines matching {} for {} in {}".format(self._matching, use, self._spline_folder))
                continue
            elif not len(spline_subset)==1:
                logging.warn("Did not find exacly one spline matching {} for {} in {}".format(self._matching, use, self._spline_folder))
                continue

            self._splinetable[use] = ps.SplineTable(spline_subset[0])


        for container in self.data:
            container["domeff_splines"] = np.ones(container.size)
            container["domeff_cache"] = np.ones(container.size)

            if container.size==0:
                continue
            
            if container.name in ["numu_cc", "numubar_cc"]:
                use = "track"
            else:
                use = "shower"     

            casc_mask = container["pid"]==0
            track_mask = np.logical_not(casc_mask)       

            if len(container["reco_energy"][casc_mask])!=0:
                container["domeff_cache"][casc_mask] = self._splinetable["shower"].evaluate_simple((
                    np.log10(container["reco_energy"][casc_mask]),
                    container["reco_coszen"][casc_mask],
                    [self._central_domeff]
                    ))
            if len(container["reco_energy"][track_mask])!=0:
                container["domeff_cache"][track_mask] = self._splinetable["track"].evaluate_simple((
                    np.log10(container["reco_energy"][track_mask]),
                    container["reco_coszen"][track_mask],
                    [self._central_domeff]
                    ))

    @profile
    def compute_function(self):


        for container in self.data:
            if container.size==0:
                continue

            casc_mask = container["pid"]==0
            track_mask = np.logical_not(casc_mask)

            if len(container["reco_energy"][casc_mask])!=0:
                casc_rate = self._splinetable["shower"].evaluate_simple((
                    np.log10(container["reco_energy"][casc_mask]),
                    container["reco_coszen"][casc_mask],
                    [self.params.domeff.value.m_as("dimensionless")]
                ))
                scales = np.power(10.0, casc_rate - container["domeff_cache"] )
                container["domeff_splines"][casc_mask] = scales

            if len(container["reco_energy"][track_mask])!=0:
                track_rate = self._splinetable["track"].evaluate_simple((
                    np.log10(container["reco_energy"][track_mask]),
                    container["reco_coszen"][track_mask],
                    [self.params.domeff.value.m_as("dimensionless")]
                ))

                scales = np.power(10.0, track_rate - container["domeff_cache"] )
                container["domeff_splines"][track_mask] = scales


    
    def apply_function(self):
        for container in self.data:
            container["weights"] = container["weights"]*container["domeff_splines"]

            container["weights"][container["weights"]< 0] = 0.0