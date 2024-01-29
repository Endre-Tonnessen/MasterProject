import pandas as pd
import numpy as np

scale = 9.941747573

def get_coords(granule_fourier2: pd.DataFrame):
    """Calculates the exact border coordinates for the given granule in an image

    Args:
        granule_fourier2 (pd.DataFrame): Granule in datarame

    Returns:
        Two lists, xs and ys, containing coords
    """
    magnitude2 = granule_fourier2['magnitude']
    # Used for getting relative coords for plotting granules in their correct positions in an image
    # bbox_left2 = granule_fourier2['bbox_left'].iloc[0]
    # bbox_right2 = granule_fourier2['bbox_right'].iloc[0]
    # bbox_top2 = granule_fourier2['bbox_top'].iloc[0]
    # bbox_bottom2 = granule_fourier2['bbox_bottom'].iloc[0]

    mean_radius2 = granule_fourier2['mean_radius'].iloc[0]
    x_pos2 = granule_fourier2['x'].iloc[0]
    y_pos2 = granule_fourier2['y'].iloc[0]
    
    # Used for getting relative coords for plotting granules in their correct positions in an image
    # x_pos_relative2 = x_pos2 - bbox_left2
    # y_pos_relative2 = y_pos2 - bbox_bottom2

    angles2 = np.linspace(0,2*np.pi,400) # sample 400 angles, as in boundary_extraction.py
    magnitudes_2 = np.append( np.array([0.0+0.0j, 0.0+0.0j]), magnitude2) * 400
    radii_12 = mean_radius2 + np.fft.irfft(magnitudes_2, 400)

    # Float coords for boundry
    ys = scale*radii_12*np.cos(angles2)+x_pos2
    xs = scale*radii_12*np.sin(angles2)+y_pos2
    return xs, ys