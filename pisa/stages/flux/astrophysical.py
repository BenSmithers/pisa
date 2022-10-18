"""
stage to implement getting the contribution to fluxes from astrophysical neutrino sources
"""
import numpy as np
from math import log10

from pisa.utils.profiler import profile
from pisa import FTYPE, TARGET, PISA_NUM_THREADS
from pisa import ureg

from pisa.core.stage import Stage

from pisa.utils.log import logging

from numba import njit, prange  # trivially parallelize for-loops


try:
    import nuSQUIDSpy as nsq
except ImportError:
    import nuSQuIDS as nsq


PIVOT = FTYPE(1.0e5)


class astrophysical(Stage):
    """
    Stage to weight the pipeline to an astrophysical flux. 
    This stage stands in place of a typical (flux + osc) stage combo


    Parameters
    ----------
    params
        Expected params are .. ::
            astro_delta : quantity (dimensionless)
            astro_norm : quantity (dimensionless)

    TODO: Add more astrophysical flux implementations
        - Broken power law
        - whatever else is needed... 
    """

    def __init__(self, 
            e_ratio = 1.0,
            mu_ratio = 1.0,
            tau_ratio = 1.0,
            rel_err=None,
            abs_err=None,
            **std_kwargs):

        self.num_neutrinos = 4 #flavors 

        self._central_gamma = FTYPE(-2.5)
        self._central_norm = FTYPE(1.0e-18)

        self._e_ratio = FTYPE(e_ratio)
        self._mu_ratio = FTYPE(mu_ratio)
        self._tau_ratio = FTYPE(tau_ratio)

        self.rel_err = rel_err.m_as("dimensionless") if rel_err is not None else 1.0e-6
        self.abs_err = abs_err.m_as("dimensionless") if abs_err is not None else 1.0e-6
        self.concurrent_threads = PISA_NUM_THREADS if TARGET == "parallel" else 1

        expected_params = ("astro_delta", "astro_norm")

        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

    def get_initial_state(self)->np.ndarray:
        """ 
        The initial state should be a power-law flux with the relevant flavor ratios! 
        """
        flavors = self.num_neutrinos 
        neutrinos = 2
        inistate = np.zeros(shape=(flavors, neutrinos,len(self.zeniths),  len(self.energies) ))

        for i_flav in range(flavors):
            for j_nu in range(neutrinos):
                _energies, _zeniths = np.meshgrid(self.energies , self.zeniths)

                inistate[i_flav][j_nu] = self._central_norm*np.power((_energies/(1e9)) / PIVOT, self._central_gamma)

        return np.transpose(inistate, axes=(2, 3, 1, 0))

    def setup_squid(self):
        """
        Sets up a nusquids object for astrophysical fluxes 
        The main goal of this is to apply attenuation to the astrophysical flux 
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
        this_osc.Set_IncludeOscillations(False) # NO OSCILLATIONS

        this_osc.Set_MixingAngle(0,1,0.563942)
        this_osc.Set_MixingAngle(0,2,0.154085)
        this_osc.Set_MixingAngle(1,2,0.785398)
        this_osc.Set_SquareMassDifference(1,7.65e-05)
        this_osc.Set_SquareMassDifference(2,0.00247)

        this_osc.Set_TauRegeneration(True)

        this_osc.Set_GSL_step(nsq.GSL_STEP_FUNCTIONS.GSL_STEP_RK4)

        return this_osc

    def setup_function(self):
        """
        Setup the nominal flux
        """
        logging.debug("seting up astro stage")
        self.data.representation = self.calc_mode
        for container in self.data:
            container["astro_flux_nominal"] = np.ones(container.size, dtype=FTYPE)

        nsq_units = nsq.Const()

        min_e = 10*nsq_units.GeV
        max_e = 100*nsq_units.PeV

        e_nodes = 100
        cth_nodes = 80

        self.energies = np.logspace(log10(min_e), log10(max_e), e_nodes)
        self.zeniths = np.linspace(-1, 1, cth_nodes)

        self.squid_atm = self.setup_squid()
        self.squid_atm.Set_initial_state(self.get_initial_state(), nsq.Basis.flavor)
        self.squid_atm.EvolveState()

        # Loop over containers
        for container in self.data:
            scale_e = container["true_energy"]*(1e9)

            for i_evt in range(container.size):
                i_nu = 0 if container["nubar"] < 0 else 1
                container["astro_flux_nominal"][i_evt] = self.squid_atm.EvalFlavor( container["flav"], container["true_coszen"][i_evt],  scale_e[i_evt], i_nu )


            # TODO split this up so that we can use flavor ratios
            # nu_flux_nominal[:,0] = _precalc*self._e_ratio
            # nu_flux_nominal[:,1] = _precalc*self._mu_ratio
            # nu_flux_nominal[:,2] = _precalc*self._tau_ratio

            container.mark_changed("astro_flux_nominal")


    @profile
    def apply_function(self):
        for container in self.data:
            if container["flav"]==0:
                scale = self._e_ratio
            elif container["flav"]==1:
                scale = self._mu_ratio
            elif container["flav"]==2:
                scale = self._tau_ratio
            else:
                logging.warn(container.name)
                logging.fatal("Unknown flavor {}".format(container["flav"]))

            if container.size==0:
                continue


            container["weights"] = container["weights"] * container["astro_flux_nominal"] * scale \
                * self.params.astro_norm.value.m_as("dimensionless") \
                * np.power(container["true_energy"]/PIVOT, self.params.astro_delta.value.m_as("dimensionless"))

            container.mark_changed("weights")
