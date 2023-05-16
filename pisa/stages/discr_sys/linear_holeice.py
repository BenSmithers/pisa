import numpy as np
import h5py as h5 

from math import log10

from scipy.stats import binned_statistic_2d

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.core.binning import MultiDimBinning
from pisa.utils.log import logging
from pisa.utils.resources import find_resource

from scipy.interpolate import RectBivariateSpline
"""
We need to do two things! 

Weight the cc-numu and cc-nunubar one way and the rest another 
"""

class linear_holeice(Stage):
    def __init__(
        self, 
        scales_file,
        **std_kwargs):

        self.scales_files = find_resource(scales_file)

        self._energy_bins = np.array([])
        self._czen_bins = np.array([])

        self._scales_energy_0 = np.array([])
        self._scales_czen_0 = np.array([])
        self._scales_energy_1 = np.array([])
        self._scales_czen_1 = np.array([])

        expected_params=[
            "holeice_p0",
            "holeice_p1"
        ]

        super().__init__(
            expected_params=expected_params,
            **std_kwargs
        )


    
    def load_scales(self)->bool:
        data = h5.File(self.scales_files,'r')

        self._energy_bins = np.array(data["energy_bins"])
        loge = np.log10(self._energy_bins)
        e_c = 0.5*(loge[:-1] + loge[1:])

        self._czen_bins = np.array(data["coszen_bins"][:])
        c_c = 0.5*(self._czen_bins[:-1] + self._czen_bins[1:])

        self._scales_0 = RectBivariateSpline(c_c, e_c,np.array(data["holeice_0"][:]))
        self._scales_1 = RectBivariateSpline(c_c, e_c,np.array(data["holeice_1"][:]))

        data.close()

        return True
        
    def setup_function(self):
        """
            Figure out in which bin all the events belong, assign the scaling parameter
        """

        
        logging.debug("setting up linear hole ice stage")
        self.load_scales()

        for container in self.data:
            # SEE np.digitize
            
            container["effect"] = np.ones(shape=len(container["weights"]))

            if len(container["effect"])==0:
                
                continue


            container["h0_effect"] = self._scales_0(container["reco_coszen"], np.log10(container["reco_energy"]), grid=False)
            container["h1_effect"] = self._scales_1(container["reco_coszen"], np.log10(container["reco_energy"]), grid=False)


            #container["h0_effect"] = self._scales_energy_0[np.digitize(container["reco_energy"], self._energy_bins)] + self._scales_czen_0[np.digitize(container["reco_coszen"], self._czen_bins)]
            #container["h1_effect"] = self._scales_energy_1[np.digitize(container["reco_energy"], self._energy_bins)] +  self._scales_czen_1[np.digitize(container["reco_coszen"], self._czen_bins)]

    def compute_function(self):
        for container in self.data:
            if len(container["effect"]) == 0:
                continue
            container["effect"] = 1 + self.params.holeice_p0.value.m_as("dimensionless")*container["h0_effect"]
            container["effect"] *= 1 + self.params.holeice_p1.value.m_as("dimensionless")*container["h1_effect"]

            container["effect"][container["effect"]<0] = 0.0

            container.mark_changed("effect")

    def apply_function(self):


        for container in self.data:
            container["weights"]*=container["effect"]

            container.mark_changed("weights")