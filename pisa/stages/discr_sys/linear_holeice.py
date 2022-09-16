import numpy as np
import h5py as h5 

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.core.binning import MultiDimBinning
from pisa.utils.log import logging
from pisa.utils.resources import find_resource

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

        self._energy_bins = np.array(data["energy_bins"][:])
        self._czen_bins = np.array(data["coszen_bins"][:])

        self._scales_energy_0 = np.array(data["holeice_0_energy"][:])
        self._scales_czen_0 = np.array(data["holeice_0_coszen"][:])
        self._scales_energy_1 = np.array(data["holeice_1_energy"][:])
        self._scales_czen_1 = np.array(data["holeice_1_coszen"][:])

        data.close()

        return True
        
    def setup_function(self):
        """
            Figure out in which bin all the events belong, assign the scaling parameter
        """
        logging.debug("setting up linear hole ice stage")
        self.load_scales()

        success = self.load_scales()
        if not success:
            raise RuntimeError("Failed to load scales file")


        for container in self.data:
            # SEE np.digitize



            container["h0_energy_scale"] = self._scales_energy_0[np.digitize(container["true_energy"], self._energy_bins[:-1])]
            container["h0_czen_scale"] = self._scales_czen_0[np.digitize(container["true_coszen"], self._czen_bins[:-1])]
            container["h1_energy_scale"] = self._scales_energy_1[np.digitize(container["true_energy"], self._energy_bins[:-1])]
            container["h1_czen_scale"] = self._scales_czen_1[np.digitize(container["true_coszen"], self._czen_bins[:-1])]

            container.mark_changed("h0_energy_scale")
            container.mark_changed("h0_czen_scale")
            container.mark_changed("h1_energy_scale")
            container.mark_changed("h1_czen_scale")

    def apply_function(self):


        for container in self.data:
            container["weights"]*= (1.0+self.params.holeice_p0.value.m_as("dimensionless")*container["h0_energy_scale"])
            container["weights"]*= (1.0+self.params.holeice_p0.value.m_as("dimensionless")*container["h0_czen_scale"])

            container["weights"]*= (1.0+self.params.holeice_p1.value.m_as("dimensionless")*container["h1_energy_scale"])
            container["weights"]*= (1.0+self.params.holeice_p1.value.m_as("dimensionless")*container["h1_czen_scale"])

            container.mark_changed("weights")