"""
stage to implement getting the contribution to fluxes from astrophysical neutrino sources
"""
import numpy as np
from math import log10, pi
import pickle

from pisa.utils.profiler import profile
from pisa import FTYPE, TARGET, PISA_NUM_THREADS
from pisa import ureg
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.correlated_param_stage import correlated_stage
from pisa.utils.resources import find_resource

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
            spline_file = "",
            **std_kwargs):

        self.num_neutrinos = 3 #flavors 

        self._central_gamma = FTYPE(-2.5)
        self._central_norm = FTYPE(1.0e-18)

        self._e_ratio = FTYPE(e_ratio)
        self._mu_ratio = FTYPE(mu_ratio)
        self._tau_ratio = FTYPE(tau_ratio)

        self.rel_err =  1.0e-6
        self.abs_err =  1.0e-6
        self.concurrent_threads = PISA_NUM_THREADS if TARGET == "parallel" else 1

        self._spline_file = spline_file if spline_file=="" else find_resource(spline_file)
        self._power_law_ini = spline_file==""

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
        neutrinos = 2 #nu/nubar
        
        if self._power_law_ini:
            inistate = np.zeros(shape=(flavors, neutrinos,len(self.zeniths),  len(self.energies) ))
            for i_flav in range(flavors):
                for j_nu in range(neutrinos):
                    _energies, _zeniths = np.meshgrid(self.energies , self.zeniths)

                    inistate[i_flav][j_nu] = self._central_norm*np.power((_energies/(1e9)) / PIVOT, self._central_gamma)

            return np.transpose(inistate, axes=(2, 3, 1, 0))
        else:
            gpsplinefile = np.load(self._spline_file, allow_pickle=True)
            try:
                gpspline = gpsplinefile["spline"].item()
            except pickle.UnpicklingError as e:
                print("Pickle problem!")
                print("Odds are the file was pickled using a different version of scipy.")
                print(e)

            inistate =  np.ones(shape=(len(self.zeniths), len(self.energies), 2, 3 ))
            loges = np.log10(self.energies/(1e9))
            pure_zen = np.arccos(self.zeniths)

            for izen in range(len(self.zeniths)):
                for ie in range(len(self.energies)):
                    flux = self._central_norm*((self.energies[ie]/(1e9))/PIVOT)**(self._central_gamma)

                    azimuth = np.random.rand()*2*pi - pi
                    if loges[ie]>7.95 or loges[ie]<1:
                        inistate[izen][ie]*= flux
                    else:
                        flux_eval = 10**gpspline((loges[ie], pure_zen[izen], azimuth))
                        inistate[izen][ie] *= (flux + flux_eval)

            return inistate


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
                i_nu = 1 if container["nubar"] < 0 else 0
                container["astro_flux_nominal"][i_evt] = self.squid_atm.EvalFlavor( container["flav"], container["true_coszen"][i_evt],  scale_e[i_evt], i_nu )


            container.mark_changed("astro_flux_nominal")
            container["astro_flux_nominal"][container["astro_flux_nominal"] < 0] = 0.0 
            # TODO split this up so that we can use flavor ratios
            # nu_flux_nominal[:,0] = _precalc*self._e_ratio
            # nu_flux_nominal[:,1] = _precalc*self._mu_ratio
            # nu_flux_nominal[:,2] = _precalc*self._tau_ratio

    def compute_function(self):
        for container in self.data:
            if container.size==0:
                continue

            if container["flav"]==0:
                scale = self._e_ratio
            elif container["flav"]==1:
                scale = self._mu_ratio
            elif container["flav"]==2:
                scale = self._tau_ratio
            else:
                logging.warn(container.name)
                logging.fatal("Unknown flavor {}".format(container["flav"]))

            container["astro_effect"] =  scale \
                * self.params.astro_norm.value.m_as("dimensionless") \
                * np.power(container["true_energy"]/PIVOT, self.params.astro_delta.value.m_as("dimensionless"))
            container["astro_effect"][container["astro_effect"]<0]=0.0

    @profile
    def apply_function(self):
        for container in self.data:
            if container.size==0:
                continue


            container["weights"] = container["weights"] * container["astro_flux_nominal"] * container["astro_effect"]

PIVOT = FTYPE(1.0e5)
