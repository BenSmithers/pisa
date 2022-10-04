
from multiprocessing.sharedctypes import Value
import numpy as np
import h5py as h5

from pisa import FTYPE
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.resources import find_resource
from scipy.interpolate import RectBivariateSpline

from pisa.utils.profiler import profile


class h5flux(Stage):
    def __init__(self,
            fluxfile:str,
            convmode=True,
            **std_kwargs):

        self.fluxfile = find_resource(fluxfile)

        self._convmode = convmode

        self.n_nu = 3

        super().__init__(
            expected_params=[],
            **std_kwargs
        )


    def setup_function(self):
        
        self._squid_flux = h5.File(self.fluxfile, 'r')
        
        energy_nodes = np.array(self._squid_flux["energy_nodes"][:])
        cos_nodes = np.array(self._squid_flux["costh_nodes"][:])

        self.fluxes = {}
        for key in self._squid_flux.keys():
            if key=="energy_nodes" or key=="costh_nodes":
                continue
            if self._convmode and "pr" in key:
                continue
            self.fluxes[key] = RectBivariateSpline(cos_nodes, np.log10(energy_nodes), np.array(self._squid_flux[key][:]))

        logging.info("Logged flux keys: {}".format(self.fluxes.keys()))

        self.data.representation = self.calc_mode
        if self.data.is_map:

            self.data.link_containers('nue',['nue_cc','nue_nc'])
            self.data.link_containers('nuebar',['nuebar_cc','nuebar_nc'])
            self.data.link_containers('numu',['numu_cc','numu_nc'])
            self.data.link_containers('numubar',['numubar_cc','numubar_nc'])    
            self.data.link_containers('nutau',['nutau_cc','nutau_nc'])
            self.data.link_containers('nutaubar',['nutaubar_cc','nutaubar_nc'])


        for container in self.data:
            container['nu_flux_nominal'] = np.empty((container.size, self.n_nu), dtype=FTYPE)
            container['nubar_flux_nominal'] = np.empty((container.size, self.n_nu), dtype=FTYPE)
            # container['nu_flux'] = np.empty((container.size, 2), dtype=FTYPE)

        # don't forget to un-link everything again
        self.data.unlink_containers()

    @profile
    def compute_function(self):
        self.data.representation = self.calc_mode
        if self.data.is_map:

            self.data.link_containers('nue',['nue_cc','nue_nc'])
            self.data.link_containers('nuebar',['nuebar_cc','nuebar_nc'])
            self.data.link_containers('numu',['numu_cc','numu_nc'])
            self.data.link_containers('numubar',['numubar_cc','numubar_nc'])    
            self.data.link_containers('nutau',['nutau_cc','nutau_nc'])
            self.data.link_containers('nutaubar',['nutaubar_cc','nutaubar_nc'])

        for container in self.data:
            key = ""
            if self._convmode:
                key = "conv_"
            else:
                key = "pr_"

            if "bar" in container.name:
                fluxkey = "nubar_flux_nominal"
                key+="antinu"
            else:
                fluxkey = "nu_flux_nominal"
                key+="nu"

            if "e" in container.name:
                key+="e"
                index=0
                
            elif "mu" in container.name:
                key+="mu"
                index=1
            elif "tau" in container.name:
                key+="tau"
                index=2
            else:
                raise Exception()

            #container[ nuflux/nubarflux ][flavor]
            
            flux = np.zeros((3, container.size),dtype=FTYPE)
            flux[index] += self.fluxes[key](
                container["true_coszen"], 
                np.log10(container["true_energy"]),
                grid=False)

            container[fluxkey] += np.transpose(flux) 

            container.mark_changed(fluxkey)

        self.data.unlink_containers()