import pandas as pd
import numpy as np

def get_coords(granule_fourier2: pd.DataFrame, get_relative=False, scale_factor=1):
    """Calculates the exact border coordinates for the given granule in an image

    Args:
        granule_fourier2 (pd.DataFrame): Granule in datarame
        get_relative (bool): If function should return coords relative to the granules bounding box and not the entire image. Useful for plottin granule cutouts from an image frame.
        scale_factor (int): Scale the boundry. Default is 1. Useful for getting boundries of upsized granule cutouts.
    Returns:
        Two lists, xs and ys, containing coords of boundry
    """
    scale = 9.941747573 # Granule scaling factor from analyzed boundry data to correct image size
    
    magnitude2 = granule_fourier2['magnitude']
    # Used for getting relative coords for plotting granules in their correct positions in an image
    bbox_left2 = granule_fourier2['bbox_left'].iloc[0]
    # bbox_right2 = granule_fourier2['bbox_right'].iloc[0]
    # bbox_top2 = granule_fourier2['bbox_top'].iloc[0]
    bbox_bottom2 = granule_fourier2['bbox_bottom'].iloc[0]

    mean_radius2 = granule_fourier2['mean_radius'].iloc[0]
    x_pos2 = granule_fourier2['x'].iloc[0]
    y_pos2 = granule_fourier2['y'].iloc[0]
    
    # Used for getting relative coords for plotting granules in their correct positions in an image
    x_pos_relative2 = x_pos2 - bbox_left2
    y_pos_relative2 = y_pos2 - bbox_bottom2

    angles2 = np.linspace(0,2*np.pi,400) # sample 400 angles, as in boundary_extraction.py
    magnitudes_2 = np.append( np.array([0.0+0.0j, 0.0+0.0j]), magnitude2) * 400
    radii_12 = mean_radius2 + np.fft.irfft(magnitudes_2, 400) 

    # Where to position the granule relative to image
    if get_relative: 
        # Position relative to bounding box
        ys = scale*radii_12*np.cos(angles2)+x_pos_relative2
        xs = scale*radii_12*np.sin(angles2)+y_pos_relative2
        return xs, ys
    else:
        # Absolute position in image 
        ys = scale*radii_12*np.cos(angles2)+x_pos2
        xs = scale*radii_12*np.sin(angles2)+y_pos2
        return xs, ys

def pixels_between_points(xs: list[float], ys: list[float], precision: int = 5, scale_factor_x=1, scale_factor_y=1) -> list[list[int],list[int]]:
    """
        From two lists, x and y-coords, returns every pixel intersected by the linesegments between the coordinates.

    Args:
        xs (list[float]): x-coords
        ys (list[float]): y-coords
        precision (int): Amount of points in the linspace between two coordinates.
        scale_factor (int): Scales the coordinates by an int. Default is 1.
    Returns:
        Two lists, xs and ys containing coordinates for every intersected pixel.
    """
    assert len(xs) == len(ys), f"Coordinate lists must be of equal length, was xs={len(xs)}, ys={len(ys)}"
    
    # Phi, the offset introduced by scaling the image. Error due to subpixel math. TODO: Look more into this. Explain why this happens.
    offset_push_x = scale_factor_x / 2 - 1/2
    offset_push_y = scale_factor_y / 2 - 1/2 
    # Scale coordinates
    xs = offset_push_x+np.array(xs)*scale_factor_x
    ys = offset_push_y+np.array(ys)*scale_factor_y

    x_pixels = np.array([])
    y_pixels = np.array([])

    for i in range(len(xs)-1):
        x_0 = xs[i]
        y_0 = ys[i]
        x_1 = xs[i+1]
        y_1 = ys[i+1]

        x_space = np.linspace(x_0,x_1, precision) #        x_space = np.round(np.linspace(x_0,x_1, precision),0)
        y_space = np.linspace(y_0,y_1, precision) #        y_space = np.round(np.linspace(y_0,y_1, precision),0)
        x_pixels = np.append(x_pixels,x_space)
        y_pixels = np.append(y_pixels,y_space)

    return np.round(x_pixels,0), np.round(y_pixels,0) # Do floor instead. Remove 1/2 from the phi offset.
    # 