"""
Neutrino/Anti-Neutrino Weighter
"""
from pisa.core.stage import Stage

class neuaneu(Stage):
    def __init__(self, **kwargs):
        expected_params = [ 
            "neuaneu_ratio"
        ]

        super().__init__(
            expected_params=expected_params,
            **kwargs
        )

    def apply_function(self):
        """
            1 = equal weighting of particles and antiparticles,
            0 = zero weight for particles, double weight for antiparticles
            2 = double weight for particles, zero weight for antiparticles
        """
        for container in self.data:
            scale = self.params.neuaneu_ratio.value.m_as("dimensionless")

            container["weights"] *= scale if container["nubar"]>0 else 2-scale

            container.mark_changed("weights")