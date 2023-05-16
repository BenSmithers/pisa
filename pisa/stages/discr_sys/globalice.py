"""
Reweights based off the global absorption and scattering parameters 
"""
import numpy as np
import h5py as h5

from pisa.utils.resources import find_resource
from pisa.core.stage import Stage
from pisa.utils.log import logging


from scipy.interpolate import RectBivariateSpline

class globalice(Stage):
    """
    A very simple stage that implements the global absorption and scattering uncertainty.
    Relies on an hdf5 file that has 1sigma perturbation effects for each of the parameters in some 2D binned setup 
    """
    def __init__(
            self,
            ice_file,
            **std_kwargs):
        
        self.ice_file = find_resource(ice_file)

        
        self._energy_bins = np.array([])
        self._czen_bins = np.array([])

        expected_params=[
            "absorption",
            "scattering"
        ]

        super().__init__(
            expected_params=expected_params,
            **std_kwargs
        )

    def load_ice_file(self)->bool:
        data = h5.File(self.ice_file,'r')

        self._energy_bins = np.array(data["energy_bins"])
        loge = np.log10(self._energy_bins)
        e_c = 0.5*(loge[:-1] + loge[1:])

        self._czen_bins = np.array(data["coszen_bins"][:])
        c_c = 0.5*(self._czen_bins[:-1] + self._czen_bins[1:])

        self._abs = RectBivariateSpline(c_c, e_c,np.array(data["absorption"][:]))
        self._sca = RectBivariateSpline(c_c, e_c,np.array(data["scattering"][:]))

        data.close()

    def setup_function(self):
        """
            Figure out in which bin all the events belong, assign the scaling parameter
        """
        logging.debug("setting up global abs/sca ice stage")
        self.load_ice_file() 


        for container in self.data:

            container["ice_effect"] = np.ones(shape=len(container["weights"]))

            if len(container["ice_effect"])==0:
                continue


            container["absorption"] = self._abs(container["reco_coszen"], np.log10(container["reco_energy"]), grid=False)
            container["scattering"] = self._sca(container["reco_coszen"], np.log10(container["reco_energy"]), grid=False)

    def compute_function(self):
        for container in self.data:
            
            container["ice_effect"] = np.ones(shape=len(container["weights"]))

            if container.size == 0:
                continue
            container["ice_effect"] = 1 + self.params.absorption.value.m_as("dimensionless")*container["absorption"]
            container["ice_effect"] *= 1 + self.params.scattering.value.m_as("dimensionless")*container["scattering"]

            container["ice_effect"][container["ice_effect"]<0] = 0.0

           # container.mark_changed("effect")

    def apply_function(self):
        for container in self.data:
            container["weights"]*=container["ice_effect"]

          #  container.mark_changed("weights")