import pybees as pb
import numpy as np


def SimpleBeesContinuous(self):
    sbc = pb.SimpleBeesContinuous(
        n_scout_bees=50,
        elite_site_params=(15, 40),
        best_site_params=(15, 30),
        bounds=(-10, 10),
        n_dim=2,
        nbhd_radius=2
    )

    return sbc


def SimpleBeesDiscrete(list):
    sbd = pb.SimpleBeesDiscrete(
        n_scout_bees=50,
        elite_site_params=(15, 40),
        best_site_params=(15, 30),
        coordinates=list,
        global_search=0
    )

    return sbd
