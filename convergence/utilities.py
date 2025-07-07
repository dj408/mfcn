"""
Utility classes and functions.
"""
import os
import time
import numpy as np
from typing import Tuple



def get_time_min_sec(
    t_1: float, 
    t_0: float = None
) -> Tuple[float, float]:
    """
    Calculates minutes and seconds 
    elapsed between 2 timepoints, or
    for one length of time.

    Args:
        t_1: 'end' timepoint or full
            time.
        t_0: 'start' timepoint. Leave
            'None' to calculate for a
            t_1 length of time.

    Returns:
        2-tuple of floats: min and sec
        elapsed.
    """
    if t_0 is None:
        t = t_1
    else:
        t = t_1 - t_0
    t_min, t_sec = t // 60, t % 60
    return t_min, t_sec

