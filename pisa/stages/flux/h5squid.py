"""
This stage implements both the 
    - neutrino flux implementation
    - and the oscillation stage

We load in an h5 file containing the neutrino flux around the Earth. 
Then propagate it according to neutrino oscilation parameters. 
The neutrino oscillations are not variable parameters, but fixed properties of the stage 

One major note that separates this from the other osc stage? 
This includes tau regeneration; an important element of high-energy analyses

Ben Smithers
"""

import numpy as np
import h5py as h5
from math import log10

from pisa import FTYPE, TARGET, PISA_NUM_THREADS
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.resources import find_resource
from scipy.interpolate import RectBivariateSpline

from pisa.utils.profiler import profile

try:
    import nuSQuIDS as nsq
except ImportError:
    import nuSQUIDSpy as nsq

class h5squid(Stage):
    """
    Parameters:
    ----------
    th14 : theta_14 mixing angle 
    th24 : theta_24 mixing angle
    th34 : theta_34 mixing angle 
    dmsq : \Delta_{41}^{2} m mass-squared splitting (fourth to first mass state)
    fluxfile : hdf5 file containing pre-generated fluxes from MCEq. Expects key for energy nodes, cos(theta) nodes, and 2D flux table per component
    convmode : bool, specifies whether we'll access the prompt (pr) or conventional (conv) components from the file 
    rel_err : error tolerance passed to nuSQuIDS
    abds_err: error tolerance passed to nuSQuIDS
    """
    def __init__(self,
            th14:float,
            th24:float,
            th34:float,
            dmsq:float,
            fluxfile:str,
            is_nusquids=True,
            convmode=True,
            rel_err=None,
            abs_err=None,
            **std_kwargs):

        self.fluxfile = find_resource(fluxfile)
        self._convmode = convmode
        self.is_nusquids = is_nusquids

        self.num_neutrinos = 4 

        self.theta14 = FTYPE(th14)
        self.theta24 = FTYPE(th24)
        self.theta34 = FTYPE(th34)
        self.dmsq = FTYPE(dmsq) 

        assert self.num_neutrinos<5, "Only up to four neutrinos are supported"
        
        self.rel_err = rel_err.m_as("dimensionless") if rel_err is not None else 1.0e-6
        self.abs_err = abs_err.m_as("dimensionless") if abs_err is not None else 1.0e-6
        self.concurrent_threads = PISA_NUM_THREADS if TARGET == "parallel" else 1

        super().__init__(
            expected_params=[],
            **std_kwargs
        )

    def setup_function(self):
        """
        Multi-step process
            first, we load in the atmospheric-level flux files for prompt or conv
            then, we build interpolators for each flavor 
        """
        
        nsq_units = nsq.Const()

        min_e = 10*nsq_units.GeV
        max_e = 100*nsq_units.PeV

        e_nodes = 100
        cth_nodes = 80

        self.energies = np.logspace(log10(min_e), log10(max_e), e_nodes)
        self.zeniths = np.linspace(-1, 1, cth_nodes)

        
        if self.is_nusquids:
            self.squid_atm = nsq.nuSQUIDSAtm(self.fluxfile)
            
        else:
            raise NotImplementedError()

        for container in self.data:
            if container.size==0:
                continue
            name = container.name
            if 'e' in name:
                flav = nsq.NeutrinoCrossSections_NeutrinoFlavor.electron
            elif 'mu' in name:
                flav = nsq.NeutrinoCrossSections_NeutrinoFlavor.muon
            elif 'tau' in name:
                flav = nsq.NeutrinoCrossSections_NeutrinoFlavor.tau

            i_nu = nsq.NeutrinoCrossSections_NeutrinoType.antineutrino if container["nubar"] < 0 else nsq.NeutrinoCrossSections_NeutrinoType.neutrino

            container['evt_flux'] = np.zeros(container.size, dtype=FTYPE)

            scale_e = container["true_energy"]*(1e9)
            for i in range(container.size):
                container["evt_flux"][i] = self.squid_atm.EvalFlavor( flav, container["true_coszen"][i],  scale_e[i], i_nu )
            
            container.mark_changed("evt_flux")

    def apply_function(self):
        for container in self.data:
            if container.size==0:
                continue
            container["weights"] = container["weights"] * np.abs(container["evt_flux"])

