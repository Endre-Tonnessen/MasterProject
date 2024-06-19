import pandas as pd
import numpy as np
import math

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

# def pixels_between_points_OLD(xs: list[float], ys: list[float], precision: int = 5, scale_factor_x=1, scale_factor_y=1) -> list[list[int],list[int]]:
#     """
#         From two lists, x and y-coords, returns every pixel intersected by the linesegments between the coordinates.

#     Args:
#         xs (list[float]): x-coords
#         ys (list[float]): y-coords
#         precision (int): Amount of points in the linspace between two coordinates.
#         scale_factor (int): Scales the coordinates by an int. Default is 1.
#     Returns:
#         Two lists, xs and ys containing coordinates for every intersected pixel.
#     """
#     assert len(xs) == len(ys), f"Coordinate lists must be of equal length, was xs={len(xs)}, ys={len(ys)}"
    
#     # Phi, the offset introduced by scaling the image. Error due to subpixel math. TODO: Look more into this. Explain why this happens.
#     offset_push_x = scale_factor_x / 2 - 1/2
#     offset_push_y = scale_factor_y / 2 - 1/2 
#     # Scale coordinates
#     xs = offset_push_x+np.array(xs)*scale_factor_x
#     ys = offset_push_y+np.array(ys)*scale_factor_y

#     x_pixels = np.array([])
#     y_pixels = np.array([])

#     for i in range(len(xs)-1):
#         x_0 = xs[i]
#         y_0 = ys[i]
#         x_1 = xs[i+1]
#         y_1 = ys[i+1]

#         x_space = np.linspace(x_0,x_1, precision) #        x_space = np.round(np.linspace(x_0,x_1, precision),0)
#         y_space = np.linspace(y_0,y_1, precision) #        y_space = np.round(np.linspace(y_0,y_1, precision),0)
#         x_pixels = np.append(x_pixels,x_space)
#         y_pixels = np.append(y_pixels,y_space)

#     return np.round(x_pixels,0), np.round(y_pixels,0) # Do floor instead. Remove 1/2 from the phi offset.
#     # 

# ------------------------------------------------------------------

def pixels_between_points(xs: list[float], ys: list[float]) -> tuple[list[int],list[int]]:
    assert len(xs) == len(ys), f"Coordinate lists must be of equal length, was xs={len(xs)}, ys={len(ys)}"

    x_pixels = np.array([])
    y_pixels = np.array([])
    for i in range(len(xs)-1):
        x_0 = xs[i]   + 1/2 # 1/2 is added to translate points from "center" based integer coordinates, to "corner" based coordinates.
        y_0 = ys[i]   + 1/2  
        x_1 = xs[i+1] + 1/2 
        y_1 = ys[i+1] + 1/2 

        xs_intersected, ys_intersected = intersect(x_0, y_0, x_1, y_1)
        x_pixels = np.append(x_pixels, xs_intersected)
        y_pixels = np.append(y_pixels, ys_intersected)

    # return x_pixels, y_pixels  
    return x_pixels.astype(int), y_pixels.astype(int)  

def intersect(x_0, y_0, x_1, y_1):
    """Finds pixels intersected by a line created by the two input points, (x_0, y_0) and (x_1, y_1).

    Args:
        x_0 (_type_): _description_
        y_0 (_type_): _description_
        x_1 (_type_): _description_
        y_1 (_type_): _description_

    Returns:
        _type_: _description_
    """
    # https://gamedev.stackexchange.com/questions/81267/how-do-i-generalise-bresenhams-line-algorithm-to-floating-point-endpoints
    # https://jsfiddle.net/6x7t4q1o/5
    # Made by Andrew?
    # Walkthrough: https://github.com/cgyurgyik/fast-voxel-traversal-algorithm/blob/master/overview/FastVoxelTraversalOverview.md

    #Grid cells are 1.0 X 1.0.
    x = math.floor(x_0)
    y = math.floor(y_0)
    diffX = x_1 - x_0
    diffY = y_1 - y_0
    # assert not (diffX == 0 and diffY == 0), "Distance between given points are 0. "
    if (diffX == 0 and diffY == 0): # Prevent issues with points being exactly on top of eachother
        return [],[]
    sign = lambda x: math.copysign(1, x)
    stepX = sign(diffX)
    stepY = sign(diffY)

    #Ray/Slope related maths.
    #Straight distance to the first vertical grid boundary.
    xOffset =  (math.ceil(x_0) - x_0) if x_1 > x_0 else (x_0 - math.floor(x_0))
    #Straight distance to the first horizontal grid boundary.
    yOffset =  (math.ceil(y_0) - y_0) if y_1 > y_0 else (y_0 - math.floor(y_0))
    #Angle of ray/slope.
    angle = math.atan2(-diffY, diffX)
    #NOTE: These can be divide by 0's, but JS just yields Infinity! :) #-- For Python same result is achieved by using numpy types
    #How far to move along the ray to cross the first vertical grid cell boundary.
    tMaxX = np.float64(xOffset) / math.cos(angle)
    #How far to move along the ray to cross the first horizontal grid cell boundary.
    tMaxY = np.float64(yOffset) / math.sin(angle)
    #How far to move along the ray to move horizontally 1 grid cell.
    tDeltaX = np.float64(1.0) / math.cos(angle)
    #How far to move along the ray to move vertically 1 grid cell.
    tDeltaY = np.float64(1.0) / math.sin(angle)

    #Travel one grid cell at a time.
    manhattanDistance = abs(math.floor(x_1) - math.floor(x_0)) + abs(math.floor(y_1) - math.floor(y_0))


    x_pixels = []
    y_pixels = []

    t = 0
    while t <= manhattanDistance:
        t+=1

        x_pixels.append(x)
        y_pixels.append(y)

        #Only move in either X or Y coordinates, not both.
        if (abs(tMaxX) < abs(tMaxY)):
            tMaxX += tDeltaX
            x += stepX
        else:
            tMaxY += tDeltaY
            y += stepY
    return x_pixels, y_pixels

# ------------------------------------------------------------------
  

def verifyLoS(x_0, y_0, x_1, y_1): 
    # https:#gamedev.stackexchange.com/questions/81267/how-do-i-generalise-bresenhams-line-algorithm-to-floating-point-endpoints
    # answered May 1, 2020 at 9:34
    # Andrew
 
    difX = x_1 - x_0
    difY = y_1 - x_0
    dist = abs(difX) + abs(difY)

    dx = difX / dist
    dy = difY / dist

    x_pixels = []
    y_pixels = []

    i = 0 
    while i <= math.ceil(dist):
        i+=1
        x = math.floor(x_0 + dx * i)
        y = math.floor(y_0 + dy * i)
        x_pixels.append(x)
        y_pixels.append(y)
    return x_pixels, y_pixels  



# ------------------------------------------------------------------

def pixels_between_points_bresham(xs: list[float], ys: list[float]) -> tuple[list[int],list[int]]:
    assert len(xs) == len(ys), f"Coordinate lists must be of equal length, was xs={len(xs)}, ys={len(ys)}"

    x_pixels = np.array([])
    y_pixels = np.array([])

    for i in range(len(xs)-1):
        x_0 = xs[i]
        y_0 = ys[i]
        x_1 = xs[i+1]
        y_1 = ys[i+1]

        xs,ys = bres(x_0, y_0, x_1, y_1)
        x_pixels = np.append(x_pixels,xs)
        y_pixels = np.append(y_pixels,ys)

    return x_pixels, y_pixels # Do floor instead. Remove 1/2 from the phi offset.


def bres(x1,y1,x2,y2): # https:#en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    x,y = x1,y1
    dx = abs(x2 - x1)
    dy = abs(y2 -y1)
    gradient = dy/float(dx)

    if gradient > 1:
        dx, dy = dy, dx
        x, y = y, x
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    p = 2*dy - dx
    # print(f"x = {x}, y = {y}")
    # Initialize the plotting points
    xcoordinates = [x]
    ycoordinates = [y]

    for k in range(2, dx + 2):
        if p > 0:
            y = y + 1 if y < y2 else y - 1
            p = p + 2 * (dy - dx)
        else:
            p = p + 2 * dy

        x = x + 1 if x < x2 else x - 1

        # print(f"x = {x}, y = {y}")
        xcoordinates.append(x)
        ycoordinates.append(y)
    return xcoordinates, ycoordinates