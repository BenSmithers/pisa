import numpy as np
import h5py as h5
import os 

from pisa.utils.profiler import profile
from pisa import FTYPE, TARGET, PISA_NUM_THREADS
from pisa.utils.resources import find_resource
from scipy.interpolate import interp1d

import photospline

import json

from pisa.core.stage import Stage
from pisa.utils.correlated_param_stage import correlated_stage
from copy import deepcopy

class ice_gradient_full(correlated_stage):
    """
        This stage loads in an hdf5 file containing the linear gradient (like) effects for bulk-ice uncertainty on absorption and scattering lengths
        These are pre-calculated from a dedicated MC set

        Here, we load some points of that full gradient trend and do some interpolation over it such that re-binning issues doesn't cause weird striation in the net effects 

    """
    def __init__(self,
                spline_folder,
                correlation,
                **std_kwargs):
        self._spline_folder = find_resource(spline_folder)
        self._cor_file = find_resource(correlation)

        correlated_stage.__init__(self, correlation, **std_kwargs)

    def setup_function(self):
        """
           aaaaah
        """
        self._per_event_effects = []

        # load the splines! 
        splines = {}
        for key in self._effects:
            fname = os.path.join(self._spline_folder, "{}_gradient_spline.fits".format(key))
            splines[key] = photospline.SplineTable(fname)

        for container in self.data:
            if container.size==0:
                continue

            # precalculate 
            logenergy =  np.log10(container["reco_energy"])

            per_event_effect= []
            for key in self._effects:
                per_event_effect.append( splines[key].evaluate_simple([container["reco_coszen"],logenergy]))
            self._per_event_effects.append(np.array(per_event_effect))
