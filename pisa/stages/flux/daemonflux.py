"""
This stage implements the flux uncertainties from daemonflux 
It relies on already having *propagated* gradients for the daemonflux parameters 

NOTE it is very important to treat these parameters as correlated! 

TODO (for Low-E folks) implement a bool-based mode that uses 
    1. unpropagated gradients from daemonflux (directly)
    2. the pre-written oscillation stages that do on-the-fly oscillations 
this would get around the need to have all of these pre-propagated fluxes 
"""

from pisa import FTYPE 
from pisa.core.stage import Stage
from pisa.utils.resources import find_resource

import os 
import h5py as h5
from scipy.interpolate import RectBivariateSpline
import numpy as np

class daemonflux(Stage):
    """
    This implementation relies on you having already prepared several things.
    
    gradient_folder points to the folder in which the stage can find gradient files 
        ex /data/user/bsmithers/cascades_analysis_data/DDM/0.0000E+00/

    file_template is a string with a format indicator `{}`, in which the stage can substitute the parameter name, such as to find all relevant stages
        ex ddm_{}_sterile_0.000000E+00_0.000000E+00_0.000000E+00_0.000000E+00_0.000000E+00_0.000000E+00.hdf5

    if `prompt` is true, only the GSF parameters will effect the fit 

    each of these files should be hdf5 files that contain 
        1. costh_nodes, length (M)
        2. energy_nodes [GeV], length (N)
        3. a 2D grid of shape (MxN) for each of nu/nubar and e/mu/tau for the gradient calculated in each bin for this param
    """
    def __init__(
        self, 
        gradient_folder:str,
        file_template:str,
        prompt:bool,
        **std_kwargs):

        self.gradient_folder = find_resource(gradient_folder)
        self.file_template = file_template

        self._gradients = {}
        self.prompt = bool(prompt)
        print("Daemon!")
        print(self.prompt, prompt)

        self.daemon_params = [
            'le_Kplus' ,
            'le_Kminus' ,
            'le_piplus' ,
            'le_piminus' ,
            'he_Kplus',
            'he_Kminus',
            'he_piplus',
            'he_piminus',
            'he_n',
            'he_p',
            'vhe1_piplus',
            'vhe1_piminus',
            'vhe3_Kplus',
            'vhe3_Kminus',
            'vhe3_n',
            'vhe3_p',
            'vhe3_piplus',
            'vhe3_piminus',
            'GSF_1',
            'GSF_2',
            'GSF_3',
            'GSF_4',
            'GSF_5',
            'GSF_6'
            ]

        super().__init__(
            expected_params=self.daemon_params,
            **std_kwargs
        )

    def setup_function(self):
        self.data.representation = self.calc_mode


        # first we load in the datafiles and store them in a local namespace 
        gradients = {}
        for param in self.daemon_params:
            filename = os.path.join(
                self.gradient_folder,
                self.file_template.format(param)
            )

            if not os.path.exists(filename):
                print("Could not find file {}".format(filename))
                raise IOError("Could not find file when opening {}".format(param))
            gradients[param] = {}

            datafile = h5.File(filename, 'r')
            for key in datafile.keys():
                gradients[param][key] = np.array(datafile[key][:])


        # now we prepare each gradient for each container 
        for container in self.data:
            container["daemon_effect"] = np.zeros(container.size)
            for param in self.daemon_params:
            
                container[param] = np.zeros(container.size)

                if container.size==0:
                    continue
                
                key = ""
                if container["nubar"]<0:
                    key+="antinu"
                else:
                    key+="nu"
                    
                if container["flav"]==0:
                    key+="e"
                elif container["flav"]==1:
                    key+="mu"
                else:
                    key+="tau"

                            
                interp = RectBivariateSpline( gradients[param]["costh_nodes"], np.log10(gradients[param]["energy_nodes"]),gradients[param]["conv_"+key])
                container[param] = interp(container["true_coszen"], np.log10(container["true_energy"]), grid=False)
        
        # each container now has a [daemon_parma] entry that specifies its gradient 

    def compute_function(self):
        """
        Calculate the net effect of perturbing all the parameters by however much... 
        """
        for container in self.data:
            container["daemon_effect"] = np.zeros(container.size)

            for param in self.daemon_params:
                if False: # self.prompt and ("GSF" not in param):
                    continue
                    
                # we divide by the weighted aeff so the gradient is against the "initial weight" of 1.0, which we then scale back up by the container['weighted_aeff'] when we apply the aeff stage
                container["daemon_effect"] += self.params[param].value.m_as("dimensionless")*container[param]

            container.mark_changed("daemon_effect")

    def apply_function(self):
        for container in self.data:
            container["weights"] += container["daemon_effect"]
