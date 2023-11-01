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
from pisa.utils.correlated_param_stage import correlated_stage


import os 
import h5py as h5
from scipy.interpolate import RectBivariateSpline
import numpy as np



class daemonflux(correlated_stage):
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
        correlation_file:str,
        **std_kwargs):

        self.gradient_folder = find_resource(gradient_folder)
        self.file_template = file_template
        self._cor_file = find_resource(correlation_file)

        self._gradients = {}
        self.prompt = bool(prompt)


        correlated_stage.__init__(self, 
                        correlation_file, 
                        as_gradients=True,
                        **std_kwargs)
        self._as_gradients = True


    @property
    def name_root(self):
        return "daemon"

    def setup_function(self):
        self.data.representation = self.calc_mode

        self._per_event_effects = []

        # first we load in the datafiles and store them in a local namespace 
        gradients = {}
        for param in self._effects:
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
            if container.size==0:
                continue

            per_event_effect = []
            for param in self._effects:
                
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
                per_event_effect.append(interp(container["true_coszen"], np.log10(container["true_energy"]), grid=False))
            self._per_event_effects.append(np.array(per_event_effect))
        
