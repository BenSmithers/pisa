"""
This implemenents the splined hole ice that the original MEOWS analysis used
"""



from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.numba_tools import WHERE, myjit
from pisa.utils.resources import find_resource

import photospline

import numpy as np


class spline_holeice(Stage):
    """
    Parameters
    ----------
    airs_spline : spline containing the 1-sigma shifts from AIRS data

    params : ParamSet
        Must exclusively have parameters: .. ::

            scale : quantity (dimensionless)
                the scale by which the weights are perturbed via the airs 1-sigma shift
    """

    def __init__(self, 
            hole_ice_spline, 
            central_value = 1.0,
            **std_kwargs):

        self.hole_ice_spline = find_resource(hole_ice_spline)
        self._central_value = central_value
        

        expected_params = [
            "hole_ice_scale",
        ]

        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )

    def setup_function(self):
        """
        Uses the splines to quickly evaluate the 1-sigma perturbtations at each of the events
        """

        self.spline_table = photospline.SplineTable(self.hole_ice_spline)

        # consider 'true_coszen" and 'true_energy' containers
        for container in self.data:
            container["hi_value"] = np.ones(container.size, dtype=FTYPE)

            if container.size==0:
                container["hi_cache"] = np.zeros(container.size, dtype=FTYPE)
            else:
                container["hi_cache"] = self.spline_table.evaluate_simple(
                    (np.log10(container["true_energy"]), container["true_coszen"], [self._central_value,])
                )

            container.mark_changed("hi_perturbed")

    def compute_function(self):
        for container in self.data:
            if container.size==0:
                continue
            rate = self.spline_table.evaluate_simple((
                    np.log10(container["true_energy"]),
                    container["true_coszen"],
                    [self.params.hole_ice_scale.value.m_as("dimensionless")]
                ))
            
            scales = np.power(10.0, rate - container["hi_cache"])
            mask = scales < 0
            scales[mask] = 0.0
            container["hi_value"] = scales
            
            container.mark_changed("hi_value")

    
    def apply_function(self):
        for container in self.data:
            if container.size==0:
                continue
            container["weights"] = container["weights"]*container["hi_value"]