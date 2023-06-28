from pisa.core.stage import Stage
from pisa.utils.resources import find_resource
import pandas as pd
import numpy as np

from pisa.core.container import Container
from pisa import FTYPE
from pisa.utils.log import logging


try:
    import nuSQUIDSpy as nsq
except ImportError:
    import nuSQuIDS as nsq

PIVOT = FTYPE(1.0e5)


class fastmc(Stage):
    """
        This loads in the fastmode MC with pre-calculated flux weights.
        Make sure this is the right one for your tested hypothesis 

        events_file     - this is your fastMC file
        output_names    - same as for the csv_loader. 
        which_flux      - used to say whether this is a conv/prompt/astro pipe 
        fluxfile        - used to calculate how much of the weight comes from a flux. Needed for the gradient weighters. Only needed for conv! 
    """

    def __init__(self,
                 events_file,
                 output_names,
                 which_flux="",
                 fluxfile = "",
                 **std_kwargs):
        
        self.events_file = find_resource(events_file)
        print("Loading {}".format(self.events_file))

        allowed_flux = ["conv", "prompt", "astro"]
        if which_flux not in allowed_flux:
            logging.fatal("`which flux` must be one of {}".format(allowed_flux))

        self._flux = which_flux

        if self._flux == "astro":
            expected_params = ("astro_delta", "astro_norm")
        else:
            expected_params = ()

        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

        self.output_names = output_names

        if self._flux == "conv":
            self._fluxfile = find_resource(fluxfile)

    def setup_function(self):
        """
            This loads in the full MC file and the  

        """
        raw_data = pd.read_csv(self.events_file)
        if self._flux=="conv":
            this_flux = nsq.nuSQUIDSAtm(self._fluxfile)

        # create containers from the events
        for name in self.output_names:

            # make container
            container = Container(name)
            nubar = -1 if 'bar' in name else 1
            if 'e' in name:
                flav = 0
            if 'mu' in name:
                flav = 1
            if 'tau' in name:
                flav = 2

            # cut out right part
            pdg = nubar * (12 + 2 * flav)

            mask = raw_data['pdg'] == pdg
            if 'cc' in name:
                mask = np.logical_and(mask, True)
            else:
                mask = np.logical_and(mask, False)

            events = raw_data[mask]

            i_nu = 1 if nubar < 0 else 0      

            if self._flux=="conv":
                czen = np.array(events['true_coszen'][:])
                energy = np.array( events['true_energy'][:])*1e9
                flux_eval = np.array([this_flux.EvalFlavor(flav,  czen[ie],energy[ie], i_nu) for ie in range(len(events['true_energy']))])
            else:
                flux_eval  =1.0
            container['true_energy'] = events['true_energy'].values.astype(FTYPE)
            container['weighted_aeff'] = events[self._flux].values.astype(FTYPE) / flux_eval
            container['weights'] =np.ones(container.size, dtype=FTYPE)
            if self._flux=="conv":
                container['initial_weights'] =flux_eval.astype(FTYPE)
            else:
                container['initial_weights'] =np.ones(container.size, dtype=FTYPE)
            container['true_coszen'] = events['true_coszen'].values.astype(FTYPE)
            container['reco_energy'] = events['reco_energy'].values.astype(FTYPE)
            container['reco_coszen'] = events['reco_coszen'].values.astype(FTYPE)
            container['pid'] = events['pid'].values.astype(FTYPE)
      
            if "n_events" in events:
                container["n_events"] = events["n_events"].values.astype(int)
            else:
                container["n_events"] = np.ones(container.size, dtype=int)
            container.set_aux_data('nubar', nubar)
            container.set_aux_data('flav', flav)

            self.data.add_container(container)

        # check created at least one container
        if len(self.data.names) == 0:
            raise ValueError(
                'No containers created during data loading for some reason.'
            )
        
        
        
    def compute_function(self):
        """
            If this is an astrophysical pipeline, then we need to prepare some astro stuff 
        """
        if self._flux!="astro":
            return

        for container in self.data:
            if container.size==0:
                continue

            container["astro_effect"] = self.params.astro_norm.value.m_as("dimensionless") \
                * np.power(container["true_energy"]/PIVOT, self.params.astro_delta.value.m_as("dimensionless"))
            container["astro_effect"][container["astro_effect"]<0]=0.0

    def apply_function(self):
        for container in self.data:
            if container.size==0:
                continue

            container['weights'] = np.copy(container['initial_weights'])
            if self._flux == "astro":
                container["weights"] *= container["astro_effect"]

            logging.debug("weight sum {}".format(np.sum(container["weights"])))