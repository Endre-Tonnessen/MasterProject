import pandas as pd
import numpy as np
import math
from PIL import Image, ImageOps
from dataclasses import dataclass
from scipy.ndimage import filters
import skimage as ski

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
    
    
def scale_image_add_padding(original_image: Image, NEW_MAX_HEIGHT=1024, NEW_MAX_WIDTH=1024) -> np.array:
    """Upscales image to its max possible size while keeping aspect ratio. Any space left is filled with padding. 

    Args:
        original_image (Image): Image to upscale
        NEW_MAX_HEIGHT (int, optional): New max height of image. Defaults to 1024.
        NEW_MAX_WIDTH (int, optional): New max width of image. Defaults to 1024.

    Returns:
        np.array: Upscaled image with padding
    """
    # ------------------- Resize image -------------------
    cutout_height, cutout_width = np.array(original_image).shape[:2]
    max_scale_height = int(np.floor(NEW_MAX_HEIGHT / cutout_height))
    max_scale_width  = int(np.floor(NEW_MAX_WIDTH / cutout_width))
    scale_factor = min(max_scale_width, max_scale_height)
    upscaled_image = original_image.resize((cutout_width*scale_factor, cutout_height*scale_factor), resample=Image.Resampling.NEAREST)
    # ------------------- Add padding -------------------
    image_width, image_height = upscaled_image.size
    delta_w = NEW_MAX_WIDTH - image_width
    delta_h = NEW_MAX_HEIGHT - image_height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    upscaled_image = ImageOps.expand(upscaled_image, padding)
    return np.array(upscaled_image)

def scale_padding(original_image, img_dims: tuple[int,int], granule_fourier: pd.DataFrame, NEW_MAX_HEIGHT=1024, NEW_MAX_WIDTH=1024) -> tuple[np.array, np.array, np.array]:
    # ------------------- Upscale image -------------------
    cutout_height, cutout_width = img_dims
    max_scale_height = int(np.floor(NEW_MAX_HEIGHT / cutout_height))
    max_scale_width  = int(np.floor(NEW_MAX_WIDTH / cutout_width))
    scale_factor = min(max_scale_height, max_scale_width) # Max amount to scale by while keeping aspect ratio
    upscaled_image = original_image.resize((cutout_width*scale_factor, cutout_height*scale_factor), resample=Image.Resampling.NEAREST)
    # ------------------- Add padding -------------------
    # assert upscaled_image.size == (NEW_MAX_HEIGHT, NEW_MAX_WIDTH), f"New size of image is wrong. What? Was {upscaled_image.size} should be {NEW_MAX_HEIGHT, NEW_MAX_WIDTH}"
    image_width, image_height = (cutout_width*scale_factor, cutout_height*scale_factor)
    delta_w = NEW_MAX_WIDTH - image_width
    delta_h = NEW_MAX_HEIGHT - image_height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(upscaled_image, padding)
    # ------------------- Get pixel border -------------------
    xs, ys = get_coords(granule_fourier, get_relative=True)
    xs = np.append(xs,xs[0]) # Add connection from last element to start element # TODO: Error is in here somewhere. Image upscaling is correct, problem with border? Titlted?
    ys = np.append(ys,ys[0])
    # --- Scale border points ---
    # TODO: This should be uncommented
    # assert (np.max(xs) <= cutout_width) and (np.max(ys) <= cutout_height), f"{np.max(xs)} <= {cutout_width} | {np.max(ys)} <= {cutout_height}"
    xs_upscaled = xs * scale_factor + scale_factor / 2 - 1/2 
    ys_upscaled = ys * scale_factor + scale_factor / 2 - 1/2
    # --- Add padding to border points ---
    xs_upscaled += delta_w // 2
    ys_upscaled += delta_h // 2
    upscaled_width, upscaled_height = new_im.size
    assert (upscaled_width, upscaled_height) == (NEW_MAX_WIDTH, NEW_MAX_HEIGHT), f"Should be {(NEW_MAX_WIDTH, NEW_MAX_HEIGHT)} == {(upscaled_width, upscaled_height)}"
    
    # return None, xs_upscaled, ys_upscaled
    return np.array(new_im), xs_upscaled, ys_upscaled

# ------------------ Gradients of image -------------------------
@dataclass
class Kernels:
    """ Implemenation of two seperable kernels that represent a gradient estimations. """

    xx: np.ndarray
    xy: np.ndarray
    yx: np.ndarray
    yy: np.ndarray
    label: str

    def gradient_x(self, image):
        """ Apply the x kernels to get the gradient in the x direction. """
        return apply_seperable_kernel(image, self.xx, self.xy)

    def gradient_y(self, image):
        """ Apply the y kernels to get the gradient in the x direction. """
        return apply_seperable_kernel(image, self.yx, self.yy)
    
def apply_seperable_kernel(image, v_1, v_2):
    """ Apply a separable kernel G to an image where K = v_2 * v_1.

    Namely, v_1 is applied to the image first. """
    output = np.zeros_like(image)
    filter_kwargs = dict(mode="reflect", cval=0, origin=0)

    filters.correlate1d(image, v_1, 1, output, **filter_kwargs)
    filters.correlate1d(output, v_2, 0, output, **filter_kwargs)
    return output

fourth_order = Kernels(
    np.array([1, -8, 0, 8, -1]) / 12.0,
    [1],
    [1],
    np.array([1, -8, 0, 8, -1]) / 12.0,
    "Central fourth order",
)

# ------------------------------------------------------------
def scale_image_add_padding(original_image: Image, NEW_MAX_HEIGHT=1024, NEW_MAX_WIDTH=1024) -> np.array:
    """Upscales image to its max possible size while keeping aspect ratio. Any space left is filled with padding. 

    Args:
        original_image (Image): Image to upscale
        NEW_MAX_HEIGHT (int, optional): New max height of image. Defaults to 1024.
        NEW_MAX_WIDTH (int, optional): New max width of image. Defaults to 1024.

    Returns:
        np.array: Upscaled image with padding
    """
    # ------------------- Resize image -------------------
    cutout_height, cutout_width = np.array(original_image).shape[:2]
    max_scale_height = int(np.floor(NEW_MAX_HEIGHT / cutout_height))
    max_scale_width  = int(np.floor(NEW_MAX_WIDTH / cutout_width))
    scale_factor = min(max_scale_width, max_scale_height)
    upscaled_image = original_image.resize((cutout_width*scale_factor, cutout_height*scale_factor), resample=Image.Resampling.NEAREST)
    # ------------------- Add padding -------------------
    image_width, image_height = upscaled_image.size
    delta_w = NEW_MAX_WIDTH - image_width
    delta_h = NEW_MAX_HEIGHT - image_height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    upscaled_image = ImageOps.expand(upscaled_image, padding)
    return np.array(upscaled_image)

class _BoundaryExtractionGradient():
    """ Extract the boundary of granule using a directional gradient.
        this is for granule which appear as a solid blob in the microscope.

    Input parameters
    ----------

    granule: Granule
        Extract the boundary from this granule.


    Methods
    -------

    process_image
        calculate directional gradients for each pixel

    """
    def process_image(self, image, local_centre):
        """ Create a directional gradient of the image.

        This calculates the component of the gradient along the radial vector of
        the granule.

        This is much more resistant to other granules in the local area.
        Further, the maximum of the gradient is much more reliable than some
        arbitrary threshold value; while the sobel is useful for this, the use of
        an absolute value of the gradient caused problems.
        """

        x_grad, y_grad = self.calculate_gradient(image)
        x_rad, y_rad = self.get_angle_from_centre(image, local_centre)

        self.processed_image = x_grad * x_rad + y_grad * y_rad
        return self.processed_image

    def get_angle_from_centre(self, image, local_centre):
        """Return a normalised vector field of the angle from the local centre of the
        granule.
        """
        crop_width, crop_height = image.shape
        # Get a vector with the distance from the centre in the x and y directions
        yDist = np.arange(crop_width) -  local_centre[1]
        xDist = np.arange(crop_height) - local_centre[0]

        # Turn this into a field
        xx, yy = np.meshgrid(xDist, yDist)

        # Normalise to unit vectors
        mag = -np.sqrt(xx ** 2 + yy ** 2)

        return xx / mag, yy / mag

    def calculate_gradient(self, image: np.ndarray):
        """ Calculate the gradient field of the image.

        Parameters
        ----------
        image:np.ndarray
            The image to use, if none is provided then use the raw image of the
            granule.

        Returns
        -------
        np.ndarray:
            A XxYx2 array with the gradient field of the granule, the top most slice
            is in the x direction and the second the y direction.

        """
        im_smoothed = ski.filters.gaussian(image, 1.5)

        kern = fourth_order

        x_grad = kern.gradient_x(im_smoothed)
        y_grad = kern.gradient_y(im_smoothed)
        return x_grad, y_grad

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
        # TODO: This should be on
        # assert (np.max(xs_intersected) <= 1024) and (np.max(ys_intersected) <= 1024), f"Number out of bounds \n x_ints: {np.max(xs_intersected)} y_ints: {np.max(ys_intersected)} Origin ({x_0, y_0}) to ({x_1, y_1})"
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