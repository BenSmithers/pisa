from math import log10
import numpy as np

from pisa import FTYPE, TARGET, PISA_NUM_THREADS
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile, line_profile
from pisa.stages.osc.layers import Layers
from pisa.core.binning import MultiDimBinning
from pisa.utils.resources import find_resource
from pisa import ureg
try:
    import nuSQUIDSpy as nsq
except ImportError:
    import nuSQuIDS as nsq

"""
This should be an independent flux calculator using the standard nuSQuIDS 

it will calculate three different oscillation probabilities 
"""


__all__= "nusquids"
__author__="B. Smithers"

class ster_osc_simple(Stage):
    def __init__(
        self,
        th14,
        th24,
        th34,
        dmsq,
        rel_err=None,
        abs_err=None,
        **std_kwargs,
    ):

        self.theta14 = FTYPE(th14)
        self.theta24 = FTYPE(th24)
        self.theta34 = FTYPE(th34)
        self.dmsq = FTYPE(dmsq) 

        self.num_neutrinos = int(4)
        assert self.num_neutrinos<5, "Only up to four neutrinos are supported"
        
        self.rel_err = rel_err.m_as("dimensionless") if rel_err is not None else 1.0e-6
        self.abs_err = abs_err.m_as("dimensionless") if abs_err is not None else 1.0e-6
        self.concurrent_threads = PISA_NUM_THREADS if TARGET == "parallel" else 1

        expected_params=[]
        
        super().__init__(expected_params=expected_params, **std_kwargs)

    def prep_osc(self):
        """
            This prepares and returns a new nuSQuIDS atmosphere object! It sets it to our pre-set mixing stuff 
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

        #sterile parameters 

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


    def setup_function(self):
        nsq_units = nsq.Const()

        min_e = 10*nsq_units.GeV
        max_e = 100*nsq_units.PeV

        e_nodes = 100
        cth_nodes = 80

        self.energies = np.logspace(log10(min_e), log10(max_e), e_nodes)
        self.zeniths = np.linspace(-1, 1, cth_nodes)

        # initstate = 

        ini_e = np.zeros((cth_nodes, e_nodes, 2, self.num_neutrinos))
        ini_mu = np.zeros((cth_nodes, e_nodes, 2, self.num_neutrinos))
        ini_tau = np.zeros((cth_nodes, e_nodes, 2, self.num_neutrinos))
        ini_e[:,:,:,0] = 1.0
        ini_mu[:,:,:,1] = 1.0
        ini_tau[:,:,:,2] = 1.0

        # nsq.nuSQUIDSAtm(ZENITH,ENERGY*con.GeV, NUMNU, nsq.NeutrinoType.both, True)
        logging.debug("Evolving electron")
        self.prop_e = self.prep_osc()
        self.prop_e.Set_initial_state(ini_e, nsq.Basis.flavor)
        self.prop_e.EvolveState()
        logging.debug("Evolving muon")
        self.prop_mu = self.prep_osc()
        self.prop_mu.Set_initial_state(ini_mu, nsq.Basis.flavor)
        self.prop_mu.EvolveState()
        logging.debug("Evolving tau")
        self.prop_tau = self.prep_osc()
        self.prop_tau.Set_initial_state(ini_tau, nsq.Basis.flavor)
        self.prop_tau.EvolveState()


        self.data.representation = self.calc_mode


        for container in self.data:
            container["prob_e"] = np.zeros(container.size, dtype=FTYPE)
            container["prob_mu"] = np.zeros(container.size, dtype=FTYPE)
            container["prob_tau"] = np.zeros(container.size, dtype=FTYPE)

    
    def compute_function(self):
        """
        For each event we calculate 
            flux(e)*Pr(e--> i)
            flux(mu)*Pr(mu--> i)
            flux(tau)*Pr(mu--> i)

        
        """

        for container in self.data:
            scale_e = container["true_energy"]*(1e9)
            
            for i in range(container.shape[0]):
                
                i_nu = 0 if container["nubar"] < 0 else 1
                container["prob_e"][i] = self.prop_e.EvalFlavor( container["flav"], container["true_coszen"][i],  scale_e[i], i_nu )
                container["prob_mu"][i] = self.prop_mu.EvalFlavor( container["flav"], container["true_coszen"][i],  scale_e[i], i_nu )
                container["prob_tau"][i] = self.prop_tau.EvalFlavor( container["flav"], container["true_coszen"][i],  scale_e[i], i_nu )

                if (container["prob_e"][i]==0.0 and container["prob_mu"][i]==0.0 and container["prob_tau"][i]==0.0):
                    raise logging.fatal("All three probabilities are zero: this is suspicious")

            container.mark_changed("prob_e")
            container.mark_changed("prob_mu")
            container.mark_changed("prob_tau")

    @profile
    def apply_function(self):
        for container in self.data:
            scales = (
                container["nu_flux"][:, 0] * container["prob_e"]
                + container["nu_flux"][:, 1] * container["prob_mu"]
                + container["nu_flux"][:, 2] * container["prob_tau"]
            )
                
            container["weights"] = container["weights"] * scales