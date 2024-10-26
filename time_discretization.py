import numpy as np
from scipy.interpolate import interp1d
import tskit
from typing import List, Tuple

def interpolate_tmrca_per_window(
    position: List[float],
    tmrca: List[float],
    interval_start: int = 0,
    interval_end: int = 500_000,
    interval_size: int = 2000,
    num_points: int = 100
) -> np.ndarray:
    """
    Calculates average TMRCA estimate for given intervals.

    Args:
        position (List[float]): List of position values.
        tmrca (List[float]): List of TMRCA values corresponding to positions.
        interval_start (int): Start of the interval range.
        interval_end (int): End of the interval range.
        interval_size (int): Size of each interval.
        num_points (int): Number of points to use for interpolation within each interval.

    Returns:
        np.ndarray: Array of average TMRCA values for each interval.
    """
    interp_function = interp1d(position, tmrca, kind='previous', fill_value="extrapolate")
    intervals = np.arange(interval_start, interval_end, interval_size)
    
    def calculate_interval_average(start: float, end: float) -> float:
        x_vals = np.linspace(start, end, num_points)
        y_vals = interp_function(x_vals)
        return np.mean(y_vals)
    
    averages = [calculate_interval_average(start, end) 
                for start, end in zip(intervals[:-1], intervals[1:])]
    
    return np.array(averages)



def get_interpolated_tmrca_landscape(ts: tskit.TreeSequence, window_size: int) -> np.ndarray:
    """
    Calculate the interpolated TMRCA landscape for a given tree sequence.

    Args:
        ts (tskit.TreeSequence): The input tree sequence.
        window_size (int): The size of the windows for interpolation.

    Returns:
        np.ndarray: The interpolated TMRCA landscape.
    """
    def extract_tmrca_data(tree: tskit.Tree) -> Tuple[float, float, int, float]:
        left, right = tree.interval
        node = next(node for node in tree.nodes() if node not in [0, 1])
        tmrca = tree.time(node)
        return left, right, node, tmrca

    tmrca_landscape = [extract_tmrca_data(tree) for tree in ts.trees()]
    tmrca_array = np.array(tmrca_landscape)

    y_tmrca_interpolated = interpolate_tmrca_per_window(
        position=tmrca_array[:, 0],
        tmrca=tmrca_array[:, 3],
        interval_end=int(ts.sequence_length) + window_size,
        interval_size=window_size
    )
    return y_tmrca_interpolated


def discretize(sequence, population_time, from_side="right"):
    indices = np.searchsorted(population_time, sequence, side="right") - 1
    return np.clip(indices, 0, len(population_time) - 1).tolist()