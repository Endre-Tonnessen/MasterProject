import numpy as np
import pandas as pd

def get_coords(granule_fourier2: pd.DataFrame, get_relative=False, fix=True):
    """Calculates the exact border coordinates for the given granule in an image

    Args:
        granule_fourier2 (pd.DataFrame): Granule in datarame
        get_relative (bool): If function should return coords relative to the granules bounding box and not the entire image. Useful for plottin granule cutouts from an image frame.
    Returns:
        Two lists, xs and ys, containing coords of boundry
    """
    scale = 9.941747573 # Granule scaling factor from analyzed boundry data to correct image size
    
    magnitude = granule_fourier2['magnitude']
    # Used for getting relative coords for plotting granules in their correct positions in an image
    bbox_left = granule_fourier2['bbox_left'].iloc[0]
    bbox_bottom = granule_fourier2['bbox_bottom'].iloc[0]

    mean_radius = granule_fourier2['mean_radius'].iloc[0]
    x_pos = granule_fourier2['x'].iloc[0]
    y_pos = granule_fourier2['y'].iloc[0]
    
    # Used for getting relative coords for plotting granules in their correct positions in an image
    x_pos_relative = x_pos - bbox_left
    y_pos_relative = y_pos - bbox_bottom

    angles2 = np.linspace(0,2*np.pi,400) # sample 400 angles, as in boundary_extraction.py
    if fix:
        order_1 = granule_fourier2['order_1'].iloc[0]
        magnitudes_2 = np.append( np.array([0.0+0.0j, order_1]), magnitude) * 400
    else:
        raise Exception("Running without fix! Is this correct?")
        magnitudes_2 = np.append( np.array([0.0+0.0j, 0.0+0.0j]), magnitude) * 400
    # order_1 = granule_fourier2['order_1'].iloc[0]
    # magnitudes_2 = np.append( np.array([0.0+0.0j, order_1]), magnitude) * 400
    radii_12 = mean_radius + np.fft.irfft(magnitudes_2, 400)  

    # Where to position the granule relative to image
    if get_relative: 
        # Position relative to bounding box
        ys = scale*radii_12*np.cos(angles2) + x_pos_relative
        xs = scale*radii_12*np.sin(angles2) + y_pos_relative
        return xs, ys
    else:
        # Absolute position in image 
        ys = scale*radii_12*np.cos(angles2)+x_pos
        xs = scale*radii_12*np.sin(angles2)+y_pos
        return xs, ys
    
# def get_coords(granule_fourier2: pd.DataFrame, get_relative=False): # OLD: DEPRECATED
#     """Calculates the exact border coordinates for the given granule in an image

#     Args:
#         granule_fourier2 (pd.DataFrame): Granule in datarame
#         get_relative (bool): If function should return coords relative to the granules bounding box and not the entire image. Useful for plottin granule cutouts from an image frame.
#         REMOVE: scale_factor (int): Scale the boundry. Default is 1. Useful for getting boundries of upsized granule cutouts.
#     Returns:
#         Two lists, xs and ys, containing coords of boundry
#     """
#     scale = 9.941747573 # Granule scaling factor from analyzed boundry data to correct image size
    
#     magnitude2 = granule_fourier2['magnitude']
#     # Used for getting relative coords for plotting granules in their correct positions in an image
#     bbox_left2 = granule_fourier2['bbox_left'].iloc[0]
#     # bbox_right2 = granule_fourier2['bbox_right'].iloc[0]
#     # bbox_top2 = granule_fourier2['bbox_top'].iloc[0]
#     bbox_bottom2 = granule_fourier2['bbox_bottom'].iloc[0]

#     mean_radius2 = granule_fourier2['mean_radius'].iloc[0]
#     x_pos2 = granule_fourier2['x'].iloc[0]
#     y_pos2 = granule_fourier2['y'].iloc[0]
    
#     # Used for getting relative coords for plotting granules in their correct positions in an image
#     x_pos_relative2 = x_pos2 - bbox_left2
#     y_pos_relative2 = y_pos2 - bbox_bottom2

#     angles2 = np.linspace(0,2*np.pi,400) # sample 400 angles, as in boundary_extraction.py
#     magnitudes_2 = np.append( np.array([0.0+0.0j, 0.0+0.0j]), magnitude2) * 400
#     radii_12 = mean_radius2 + np.fft.irfft(magnitudes_2, 400) 

#     # Where to position the granule relative to image
#     if get_relative: 
#         # Position relative to bounding box
#         ys = scale*radii_12*np.cos(angles2)+x_pos_relative2
#         xs = scale*radii_12*np.sin(angles2)+y_pos_relative2
#         return xs, ys
#     else:
#         # Absolute position in image 
#         ys = scale*radii_12*np.cos(angles2)+x_pos2
#         xs = scale*radii_12*np.sin(angles2)+y_pos2
#         return xs, ys