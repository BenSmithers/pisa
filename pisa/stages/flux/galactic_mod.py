"""
This file adds extra flux to a population from the galactic plane. 
"""
from pisa import FTYPE 
from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
from scipy.interpolate import RegularGridInterpolator 

import numpy as np
import pickle 
from math import pi 
import os 

class galactic_mod(Stage):
    def __init__(self, 
            spline_file = "KRAgamma_50_DNNCascade_gal.npz",
            **kwargs):

        self._filename = find_resource(spline_file)

        super().__init__(
            expected_params=[],
            **kwargs
        )
    def setup_function(self):
        """
            Load the file in, and reweight the events. 
            We shuffle the azimuths for the MC since PISA doesn't have azimuthal support 
        """

        gpsplinefile = np.load(self._filename, allow_pickle=True)

        try:
            gpspline = gpsplinefile["spline"].item()
        except pickle.UnpicklingError as e:
            print("Pickle problem!")
            print("Odds are the file was pickled using a different version of scipy.")
            print(e)
            

        for container in self.data:
            if container.size==0:
                continue

            log_energy = np.log10(container["reco_energy"])
            off_grid_mask = np.logical_or(log_energy<1, log_energy>7.95397111)


            log_energy[off_grid_mask] = 1.0
            zenith = np.arccos(container["true_coszen"])
            azimuths = np.random.rand(len(container["true_coszen"]))*2*pi - pi

            container["gp_spline_flux"] =  10**gpspline(np.stack((log_energy, zenith, azimuths), axis=1))
            container["gp_spline_flux"][off_grid_mask]=0.0

    def apply_function(self):
        for container in self.data:
            if container.size==0:
                continue
            container["weights"] += container["gp_spline_flux"]
        