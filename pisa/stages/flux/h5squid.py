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
    import nuSQUIDSpy as nsq
except ImportError:
    import nuSQuIDS as nsq

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
            convmode=True,
            separate_flux=True,
            rel_err=None,
            abs_err=None,
            **std_kwargs):

        self.fluxfile = find_resource(fluxfile)
        self._convmode = convmode

        self.num_neutrinos = 4 

        self.theta14 = FTYPE(th14)
        self.theta24 = FTYPE(th24)
        self.theta34 = FTYPE(th34)
        self.dmsq = FTYPE(dmsq) 
        self.separate_flux = separate_flux

        assert self.num_neutrinos<5, "Only up to four neutrinos are supported"
        
        self.rel_err = rel_err.m_as("dimensionless") if rel_err is not None else 1.0e-6
        self.abs_err = abs_err.m_as("dimensionless") if abs_err is not None else 1.0e-6
        self.concurrent_threads = PISA_NUM_THREADS if TARGET == "parallel" else 1

        super().__init__(
            expected_params=[],
            **std_kwargs
        )


    def setup_squid(self):
        """
        Prepare a nuSQuIDS atmosphere object we'll use for flux calculations 
        """
        this_osc = nsq.nuSQUIDSAtm(
                self.zeniths,
                self.energies,
                self.num_neutrinos,
                nsq.NeutrinoType.both,
                True # earth interactions
        )

        this_osc.Set_rel_error(self.rel_err)
        this_osc.Set_abs_error(self.abs_err)
        this_osc.Set_EvalThreads(self.concurrent_threads)
        this_osc.Set_IncludeOscillations(True)

        this_osc.Set_MixingAngle(0,1,0.563942)
        this_osc.Set_MixingAngle(0,2,0.154085)
        this_osc.Set_MixingAngle(1,2,0.785398)

        this_osc.Set_SquareMassDifference(1,7.65e-05)
        this_osc.Set_SquareMassDifference(2,0.00247)

        this_osc.Set_TauRegeneration(True)

        this_osc.Set_MixingAngle(0,3,self.theta14)
        this_osc.Set_MixingAngle(1,3,self.theta24)
        this_osc.Set_MixingAngle(2,3,self.theta34)
        this_osc.Set_SquareMassDifference(3,self.dmsq)

        this_osc.Set_GSL_step(nsq.GSL_STEP_FUNCTIONS.GSL_STEP_RK4)


        #this_osc.Set_CPPhase(0, 3, self.params.deltacp14.value.m_as("rad"))
        #this_osc.Set_CPPhase(1, 3, self.params.deltacp24.value.m_as("rad")) 

        #this_osc.Set_CPPhase(0, 2, self.params.deltacp.value.m_as("rad"))

        return this_osc

    def get_initial_state(self, only_flavor=-1)->np.ndarray:
        """
            Returns the initial state for nusquids. Assumes the energies are provided in eV
            and the zeniths as cos(zeniths)

            if `only_flavor` is provided, this will skip all flavors that aren't that flavor
        """

        flavors = self.num_neutrinos
        neutrinos = 2
        inistate = np.zeros(shape=(flavors, neutrinos,len(self.zeniths),  len(self.energies) ))
        logging.debug("shape of inistate: {}".format(np.shape(inistate)))
        for i_flav in range(flavors):
            for j_nu in range(neutrinos):
                if (only_flavor!=-1) and (only_flavor!=i_flav):
                    continue
                key = ""
                if self._convmode:
                    key = "conv_"
                else:
                    key = "pr_"

                if j_nu == 1:
                    key+="antinu"
                else:
                    key+="nu"
                
                if i_flav==0:
                    key+="e"
                elif i_flav==1:
                    key+="mu"
                elif i_flav==2:
                    key+="tau"
                else:
                    continue

                inistate[i_flav][j_nu] = np.abs(self.fluxes[key](self.zeniths, np.log10(self.energies/(1e9))))
        
        # transpose this initial state into a nuSQuIDS-friendly one
        inistate = np.transpose(inistate, axes=(2, 3, 1, 0))
        logging.debug("shape of inistate: {}".format(np.shape(inistate)))
        return inistate

    def setup_function(self):
        """
        Multi-step process
            first, we load in the atmospheric-level flux files for prompt or conv
            then, we build interpolators for each flavor 
        """
        
        self._squid_flux = h5.File(self.fluxfile, 'r')
        
        energy_nodes = np.array(self._squid_flux["energy_nodes"][:])
        cos_nodes = np.array(self._squid_flux["costh_nodes"][:])

        self.fluxes = {}
        for key in self._squid_flux.keys():
            if key=="energy_nodes" or key=="costh_nodes":
                continue
            if self._convmode and "pr" in key: #if we're in conv, skip prompt 
                continue
            if (not self._convmode) and "conv" in key: # if we're in pr, skip conv 
                continue
            self.fluxes[key] = RectBivariateSpline(cos_nodes, np.log10(energy_nodes), np.array(self._squid_flux[key][:]))

        nsq_units = nsq.Const()

        min_e = 10*nsq_units.GeV
        max_e = 100*nsq_units.PeV

        e_nodes = 100
        cth_nodes = 80

        self.energies = np.logspace(log10(min_e), log10(max_e), e_nodes)
        self.zeniths = np.linspace(-1, 1, cth_nodes)

        logging.debug("flux status: {}".format(self.separate_flux))
        if self.separate_flux:
            logging.debug("Separate flux!")
            self.e_atm = self.setup_squid()
            self.e_atm.Set_initial_state(self.get_initial_state(0), nsq.Basis.flavor)
            self.e_atm.EvolveState()

            self.mu_atm = self.setup_squid()
            self.mu_atm.Set_initial_state(self.get_initial_state(1), nsq.Basis.flavor)
            self.mu_atm.EvolveState()

            if not self._convmode: # no conventional taus...
                self.tau_atm = self.setup_squid()
                self.tau_atm.Set_initial_state(self.get_initial_state(2), nsq.Basis.flavor)
                self.tau_atm.EvolveState()
        else:
            self.squid_atm = self.setup_squid()
            self.squid_atm.Set_initial_state(self.get_initial_state(), nsq.Basis.flavor)
            self.squid_atm.EvolveState()

        self.data.representation = self.calc_mode
        if self.data.is_map:

            self.data.link_containers('nue',['nue_cc','nue_nc'])
            self.data.link_containers('nuebar',['nuebar_cc','nuebar_nc'])
            self.data.link_containers('numu',['numu_cc','numu_nc'])
            self.data.link_containers('numubar',['numubar_cc','numubar_nc'])    
            self.data.link_containers('nutau',['nutau_cc','nutau_nc'])
            self.data.link_containers('nutaubar',['nutaubar_cc','nutaubar_nc'])


        for container in self.data:
            container["flux_contribution"] = np.zeros((container.size, 3), dtype=FTYPE)
            container['evt_flux'] = np.zeros(container.size, dtype=FTYPE)

        
            scale_e = container["true_energy"]*(1e9)
            for i in range(container.shape[0]):
                
                i_nu = 0 if container["nubar"] < 0 else 1

                if self.separate_flux:
                    container["flux_contribution"][i][0] = abs(self.e_atm.EvalFlavor( container["flav"], container["true_coszen"][i],  scale_e[i], i_nu ))
                    container["flux_contribution"][i][1] = abs(self.mu_atm.EvalFlavor( container["flav"], container["true_coszen"][i],  scale_e[i], i_nu ))
                    if not self._convmode:
                        container["flux_contribution"][i][2] = abs(self.tau_atm.EvalFlavor( container["flav"], container["true_coszen"][i],  scale_e[i], i_nu ))

                    norm = np.sum(container["flux_contribution"][i])
                    container["evt_flux"][i]= 0 if norm< 1e-30 else norm

                    container["flux_contribution"][i][0] /= norm
                    container["flux_contribution"][i][1] /= norm
                    if not self._convmode:
                        container["flux_contribution"][i][2] /= norm


                else:
                    container["flux_contribution"][i][container["flav"]] = 1.0
                    container["evt_flux"][i] = self.squid_atm.EvalFlavor( container["flav"], container["true_coszen"][i],  scale_e[i], i_nu )
            
            container.mark_changed("flux_contribution")
            container.mark_changed("evt_flux")

            if (container.size!=0):
                logging.debug("Min and max {} and {}".format(np.min(container["flux_contribution"]), np.max(container["flux_contribution"])))
        
        # don't forget to un-link everything again
        self.data.unlink_containers()

    def apply_function(self):
        for container in self.data:
            container["weights"] = container["weights"] * np.abs(container["evt_flux"])

            container.mark_changed("weights")