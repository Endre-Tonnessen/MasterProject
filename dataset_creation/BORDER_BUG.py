from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from helper_functions import frame_gen as fg
from helper_functions.frame_gen import startVM, vmManager

def plot_single(full_frame, granule_cutout_image, valid_granule_id, granule_fourier: pd.DataFrame):
    xs, ys = get_coords(granule_fourier, get_relative=True)
    xs = np.append(xs,xs[0])
    ys = np.append(ys,ys[0]) 

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=granule_cutout_image, colorscale='viridis'))
    fig.add_trace(go.Scatter(x=xs, y=ys, marker=dict(color='red', size=16), name=f"400 p border {valid_granule_id}"),  )
    fig.update_layout(title_text=f"Granule {valid_granule_id}", title_x=0.5, showlegend=False, font_size=11)
    fig.update_layout(
        autosize=False,
        width=1100,
        height=1100,
    )
    fig.show()

def get_coords(granule_fourier: pd.DataFrame, get_relative=False):
    """Calculates the exact border coordinates for the given granule in an image

    Args:
        granule_fourier (pd.DataFrame): Granule in datarame
        get_relative (bool): If function should return coords relative to the granules bounding box and not the entire image. Useful for plottin granule cutouts from an image frame.
        REMOVE: scale_factor (int): Scale the boundry. Default is 1. Useful for getting boundries of upsized granule cutouts.
    Returns:
        Two lists, xs and ys, containing coords of boundry
    """
    scale = 9.941747573 # Granule scaling factor from analyzed boundry data to correct image size
    
    magnitude = granule_fourier['magnitude']
    # Used for getting relative coords for plotting granules in their correct positions in an image
    bbox_left = granule_fourier['bbox_left'].iloc[0]
    bbox_bottom = granule_fourier['bbox_bottom'].iloc[0]

    mean_radius = granule_fourier['mean_radius'].iloc[0]
    x_pos = granule_fourier['x'].iloc[0]
    y_pos = granule_fourier['y'].iloc[0]
    
    # Used for getting relative coords for plotting granules in their correct positions in an image
    x_pos_relative = x_pos - bbox_left
    y_pos_relative = y_pos - bbox_bottom

    angles = np.linspace(0,2*np.pi,400) # sample 400 angles, as in boundary_extraction.py
    magnitudes_2 = np.append( np.array([0.0+0.0j, 0.0+0.0j]), magnitude) * 400
    radii_12 = mean_radius + np.fft.irfft(magnitudes_2, 400)  

    # Where to position the granule relative to image
    if get_relative: 
        # Position relative to bounding box
        ys = scale*radii_12*np.cos(angles) + x_pos_relative
        xs = scale*radii_12*np.sin(angles) + y_pos_relative
        return xs, ys
    else:
        # Absolute position in image 
        ys = scale*radii_12*np.cos(angles)+x_pos
        xs = scale*radii_12*np.sin(angles)+y_pos
        return xs, ys

def label_and_image(ims_file_path, h5_file_path):
    image_gen = fg.bioformatsGen(Path(ims_file_path))
    image_analysed_results_df = pd.read_hdf(Path(h5_file_path), mode="r", key="fourier")

    for frame_num, frame in enumerate(image_gen):
        frame_id = frame_num
        valid_granule_fourier = image_analysed_results_df[(image_analysed_results_df['valid'] == True) & (image_analysed_results_df['frame'] == frame_id)]
        valid_granule_ids = valid_granule_fourier['granule_id'].unique()
        image_data = frame.im_data

        for valid_granule_id in valid_granule_ids: # For each valid granule in frame
            #  ------------------- Get boundry ------------------- 
            granule_fourier = valid_granule_fourier[(valid_granule_fourier['granule_id'] == valid_granule_id) & (valid_granule_fourier['frame'] == frame_id)]
            #  ------------------- Get image of granule ------------------- 
            bbox_left = granule_fourier['bbox_left'].iloc[0]
            bbox_right = granule_fourier['bbox_right'].iloc[0]
            bbox_top = granule_fourier['bbox_top'].iloc[0]
            bbox_bottom = granule_fourier['bbox_bottom'].iloc[0]
            granule_cutout_image = image_data[bbox_left:bbox_right, bbox_bottom:bbox_top]

            # if valid_granule_id == 1: #use 3,  2 is also good for testing
            if valid_granule_id == 3: #use 3,  2 is also good for testing
                plot_single(image_data, granule_cutout_image, valid_granule_id, granule_fourier)
                exit()
            
        break
        
    # Clean up
    del image_analysed_results_df
    image_gen.close()
    del image_gen

startVM()

if __name__ == "__main__":
    fg.startVM()

    @fg.vmManager
    def main():

        image_file= 'D:/Master/Granule_Explorer/granule_explorer_core/experiments/ims_files_ALL/2020-02-05_14.35.36--NAs--T1354-GFP_Burst.ims'
        h5_file_path = "D:/Master/Granule_Explorer/granule_explorer_core/experiments/gradient_all_files/fourier/2020-02-05_14.35.36--NAs--T1354-GFP_Burst.h5"
        label_and_image(image_file, h5_file_path)

    main()
