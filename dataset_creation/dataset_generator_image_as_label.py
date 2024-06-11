import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
import os
import shutil
import tqdm
import cv2
import skimage as ski
import bioformats as bf
import javabridge
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helper_functions.helper_functions import get_coords, pixels_between_points
# from dataset_creation.helper_functions.helper_functions import get_coords, pixels_between_points
from helper_functions import frame_gen as fg
from helper_functions.frame_gen import startVM, vmManager
# import dataset_creation.helper_functions.frame_gen as fg
from multiprocessing import Process, Queue
from scipy.ndimage import filters
from dataclasses import dataclass

import pathlib
import platform
plt2 = platform.system()
if plt2 == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

def plot(granule_cutout_image, upscaled_image, valid_granule_id, xs_upscaled, ys_upscaled, granule_fourier):
    fig = make_subplots(rows=1, cols=3, 
                    horizontal_spacing=0.05, 
                    vertical_spacing=0.1,
                    subplot_titles=('Standard image', 'Upscale Nearest', 'Mask fill'))
    # ------ Column 1 ------
    xs, ys = get_coords(granule_fourier, get_relative=True)
    xs = np.append(xs,xs[0])
    ys = np.append(ys,ys[0]) 
    x_centre = granule_fourier['x'].iloc[0]
    y_centre = granule_fourier['y'].iloc[0]
    bbox_left2 = granule_fourier['bbox_left'].iloc[0]
    bbox_bottom2 = granule_fourier['bbox_bottom'].iloc[0]
    x_pos_relative = x_centre - bbox_left2
    y_pos_relative = y_centre - bbox_bottom2
    fig.add_trace(go.Scatter(x=[y_pos_relative], y=[x_pos_relative], marker=dict(color='red', size=16), name=f"Centre"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=granule_cutout_image, colorscale='viridis'), row=1, col=1)

    # Calculate and draw boundry for first plot
    # xs_pixels, ys_pixels = pixels_between_points(xs, ys)
    # fig.add_trace(go.Scatter(x=xs_pixels, y=ys_pixels, marker=dict(color='cyan', size=16), name=f"Pixel border {valid_granule_id}"), row=1, col=1)

    fig.add_trace(go.Scatter(x=xs, y=ys, marker=dict(color='red', size=16), name=f"400 p border {valid_granule_id}"), row=1, col=1)

    xs_pixels, ys_pixels = pixels_between_points(xs_upscaled, ys_upscaled)
    # fig.add_trace(go.Scatter(x=xs_pixels, y=ys_pixels, marker=dict(color='cyan', size=16), name=f"Pixel border {valid_granule_id}"), row=1, col=2)
    # fig.add_trace(go.Scatter(x=xs_upscaled, y=ys_upscaled, marker=dict(color='red', size=16), name=f"400p upscaled border3 {valid_granule_id}"), row=1, col=2)
    
    # -----------------------------
    border_image = np.zeros((1024,1024))
    for i in range(len(xs_pixels)):
        border_image[ys_pixels[i], xs_pixels[i]] = 1

    import skimage as ski

    flood_fill = ski.morphology.flood(border_image, (512,512), connectivity=1)
    border_image[flood_fill == True] = 60000
    for i in range(len(xs_pixels)):
        border_image[ys_pixels[i], xs_pixels[i]] = 60000

    ### base_image_fig = px.imshow(border_image) 
    ### fig.add_trace(base_image_fig.data[0], 1, 3)
     
    # im_smoothed = ski.filters.gaussian(granule_cutout_image, 1.5)
    # fig.add_trace(go.Heatmap(z=im_smoothed, colorscale='viridis'), row=1, col=3)
    fig.add_trace(go.Heatmap(z=border_image, colorscale='viridis'), row=1, col=3)
    fig.add_trace(go.Scatter(x=xs_pixels, y=ys_pixels, marker=dict(color='cyan', size=16), name=f"Pixel border {valid_granule_id}"), row=1, col=2)
    # -----------------------------

    # upscaled_image[border_image == 60000] = 60000
    # upscaled_image_fig = px.imshow(upscaled_image)
    ### fig.add_trace(upscaled_image_fig.data[0], 1, 2)
    fig.add_trace(go.Heatmap(z=upscaled_image, colorscale='viridis'), row=1, col=2)



    im_path = granule_fourier['im_path'].iloc[0]
    im_path = Path(im_path).stem
    fig.update_layout(title_text=f"{im_path} Granule {valid_granule_id}", title_x=0.5, showlegend=False, font_size=11)
    fig.update_yaxes(autorange='reversed') # Ensure granules are not flipped. Plotly has strange axis direction defaults...
    fig.show()
    

def plot_single(full_frame, granule_cutout_image, valid_granule_id, granule_fourier: pd.DataFrame):
    xs, ys = get_coords(granule_fourier, get_relative=True)
    xs = np.append(xs,xs[0])
    ys = np.append(ys,ys[0]) 
    # ------ Centre of granule ------
    x_centre = granule_fourier['x'].iloc[0]
    y_centre = granule_fourier['y'].iloc[0]
    bbox_left2 = granule_fourier['bbox_left'].iloc[0]
    bbox_bottom2 = granule_fourier['bbox_bottom'].iloc[0]
    x_pos_relative = x_centre - bbox_left2
    y_pos_relative = y_centre - bbox_bottom2
    # ------------------------------
    # fig = go.Figure()
    # fig.add_trace(go.Heatmap(z=granule_cutout_image, colorscale='viridis'))
    # fig.add_trace(go.Scatter(x=xs, y=ys, marker=dict(color='red', size=16), name=f"400 p border {valid_granule_id}"),  )
    # fig.add_trace(go.Scatter(x=[y_pos_relative], y=[x_pos_relative], marker=dict(color='red', size=16), name=f"Centre"))
    # fig.update_layout(title_text=f"Granule {valid_granule_id}", title_x=0.5, showlegend=False, font_size=11)
    # fig.update_layout(
    # autosize=False,
    # width=1100,
    # height=1100,)
    # fig.show()
    # For saving pure image
    # plt.imsave(f"D:/Master/MasterProject/Overleaf_figures/Chapter4/Full_frames/Granule_{np.random.randint(0,1000)}_raw.png", granule_cutout_image)

    # ------------- Figure for showing the courseness of pixels in granule cutout. Limited information to segment -----------
    # xs_pixels, ys_pixels = pixels_between_points(xs, ys)
    # border_image = np.zeros(granule_cutout_image.shape)
    # for i in range(len(xs_pixels)):
    #     border_image[ys_pixels[i], xs_pixels[i]] = 1

    # fig = make_subplots(rows=1, cols=2, 
    #                 horizontal_spacing=0.05, 
    #                 vertical_spacing=0.1,)
    # # --- Col 1 --- 
    # fig.add_trace(go.Heatmap(z=granule_cutout_image, colorscale='viridis'), row=1, col=1)
    # fig.add_trace(go.Scatter(x=xs, y=ys, marker=dict(color='black', size=40), name=f"400 p border {valid_granule_id}"), row=1, col=1 )
    # # fig.add_trace(go.Scatter(x=[y_pos_relative], y=[x_pos_relative], marker=dict(color='red', size=16), name=f"Centre"), row=1, col=1)
    # # --- Col 2 --- 
    # fig.add_trace(go.Heatmap(z=border_image, colorscale='viridis'), row=1, col=2)
    # fig.add_trace(go.Scatter(x=xs, y=ys, marker=dict(color='black', size=40), name=f"400 p border {valid_granule_id}"), row=1, col=2)
    # # --- Annotation letters ---
    # fig.add_annotation(text="A",
    #               xref="paper", yref="paper",
    #               x=0.005, y=1, showarrow=False)
    # fig.add_annotation(text="B",
    #                 xref="paper", yref="paper",
    #                 x=0.54, y=1, showarrow=False)
    # fig.update_annotations(font=dict(family="DejaVu Sans", size=50, color="white"))
    
    # # --- Layout ---
    # fig.update_layout(showlegend=False, font_size=35)
    # fig.update_layout(
    #     autosize=False,
    #     width=1700,
    #     height=1000,
    # )
    # # fig.update_coloraxes(showscale=False) 
    # fig.update_layout({
    #     "paper_bgcolor": "rgba(0, 0, 0, 0)",
    #     "plot_bgcolor": "rgba(0, 0, 0, 0)", })
    # fig.update_layout(
    #     font_family="DejaVu Sans",
    #     font_color="black",
    #     title_font_family="DejaVu Sans",
    #     title_font_color="black",
    # )
    # fig.update_traces(dict(showscale=False, coloraxis=None,), selector={'type':'heatmap'}) # 2020-02-05_14.36.40--NAs--T1354-GFP_Burst | Valid granule 1
    # fig.write_image(file="D:/Master/MasterProject/Overleaf_figures/Chapter4/StressGranule_and_CourseBorder.svg", scale=4)
    # fig.show()
    # ----- The same but for only the single border - StressGranule_and_CourseBorder_1000.svg. THIS PRODUCES BOTH THE BORDER IMAGE AND AN ADDIONAL ZOOM IN VERSION OF THE BORDER
    for scale in [1,5,10]:
        cutout_width, cutout_height = granule_cutout_image.shape
        original_image = Image.fromarray(granule_cutout_image)
        granule_cutout_image_upscaled, xs, ys = scale_padding(original_image, (cutout_height, cutout_width), granule_fourier, NEW_MAX_HEIGHT = 100*scale, NEW_MAX_WIDTH = 100*scale)
        
        xs_pixels, ys_pixels = pixels_between_points(xs, ys)
        border_image = np.zeros(granule_cutout_image_upscaled.shape)
        for i in range(len(xs_pixels)):
            border_image[ys_pixels[i], xs_pixels[i]] = 1
        fig = go.Figure()
        # --- Col 1 --- 
        # fig.add_trace(go.Heatmap(z=granule_cutout_image_upscaled, colorscale='viridis'))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode = "markers", marker=dict(color='black', size=5), name=f"400 p border {valid_granule_id}") )
        # fig.add_trace(go.Scatter(x=[y_pos_relative], y=[x_pos_relative], marker=dict(color='red', size=16), name=f"Centre"),)
        # --- Col 2 --- 
        fig.add_trace(go.Heatmap(z=border_image, colorscale='viridis'))
        # --- Layout ---
        fig.update_layout(showlegend=False, font_size=55)
        fig.update_layout(
            autosize=False,
            width=1500,
            height=1000,
        )
        fig.update_layout(
                {
                    "paper_bgcolor": "rgba(0, 0, 0, 0)",
                    "plot_bgcolor": "rgba(0, 0, 0, 0)",
                }
            )
        fig.update_layout(
            font_family="DejaVu Sans",
            font_color="black",
            title_font_family="DejaVu Sans",
            title_font_color="black",
        )
        max_scale_height = int(np.floor(100*scale / cutout_height))
        max_scale_width  = int(np.floor(100*scale / cutout_width))
        scale_factor = min(max_scale_height, max_scale_width) 
        # fig.update_layout(title_text=f"Upscaled {int(scale_factor)} times", title_x=0.5, showlegend=False, font_size=20)
        fig.update_layout(showlegend=False, font_size=55)
        fig.update_traces(dict(showscale=False, coloraxis=None,), selector={'type':'heatmap'})
        fig.write_image(file=f"D:/Master/MasterProject/Overleaf_figures/Chapter4/StressGranule_and_CourseBorder_scales/StressGranule_and_CourseBorder_{100*scale}.svg", scale=4)
        fig.show()
        # --- Zoom in ---
        if scale == 1:
            yrange=[53, 63]
            xrange=[23, 29]
        elif scale == 5:
            yrange=[262, 306]
            xrange=[80, 101]
        elif scale == 10:
            yrange=[513, 715]
            xrange=[146, 250]
        layout = go.Layout(
            yaxis=dict(
                range=yrange
            ),
            xaxis=dict(
                range=xrange
            )
        )
        fig = go.Figure(layout=layout)   
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color='black', size=15), name=f"400 p border {valid_granule_id}") )
        # fig.add_trace(go.Scatter(x=[y_pos_relative], y=[x_pos_relative], marker=dict(color='red', size=16), name=f"Centre"),)
        # --- Col 2 --- 
        fig.add_trace(go.Heatmap(z=border_image, colorscale='viridis'))
        # --- Layout ---
        fig.update_layout(showlegend=False, font_size=55)
        fig.update_layout(
            autosize=False,
            width=1500,
            height=1000,
        )
        fig.update_layout({
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
            "plot_bgcolor": "rgba(0, 0, 0, 0)", })
        fig.update_layout(
            font_family="DejaVu Sans",
            font_color="black",
            title_font_family="DejaVu Sans",
            title_font_color="black",
        )
        fig.update_traces(dict(showscale=False, coloraxis=None,), selector={'type':'heatmap'})
        # fig.update_layout(title_text=f"Upscaled {int(scale_factor)} times", title_x=0.5, showlegend=False, font_size=20)
        # fig.update_yaxes(showticklabels=False)
        # fig.update_xaxes(showticklabels=False)
        fig.write_image(file=f"D:/Master/MasterProject/Overleaf_figures/Chapter4/StressGranule_and_CourseBorder_scales/StressGranule_and_CourseBorder_{100*scale}_ZOOM.svg", scale=4)




startVM()

@vmManager
def generate_granule_cutout_images(ims_file_directory_path: Path = "", 
                                   h5_analyzed_ims_data_path: Path = ""):
    """
    """
    current_file = Path(__file__).resolve() # TODO: FIX THIS PATH MESS
    project_dir = current_file.parents[1] / "dataset_creation/data/"

    image_files = [file[:-4] for file in os.listdir(f"{project_dir}\ALL_IMS")]
    # image_files= ['2020-02-05_14.35.36--NAs--T1354-GFP_Burst'] # This one
    
    # image_files= image_files[:6]
    image_files= ['2020-02-05_14.36.40--NAs--T1354-GFP_Burst']

    # processes: list[Process] = []
    image_files_queue = Queue() 
    for filename in image_files: # Populate Queue with filenames
        image_files_queue.put(filename)

    processes = [Process(target=get_label_and_image, args=(project_dir, image_files_queue,)) for _ in range(6)]

    for process in processes: 
        process.start() # Start

    for process in processes:
        process.join()  # Stop

    # process_bar_files = tqdm.tqdm(enumerate(image_files))
    # for file_num, filename in process_bar_files:
        # if file_num == 0:
        #     process_bar_files.reset(file_num)
    # for filename in image_files:
    #     p1 = Process(target=get_label_and_image, args=(project_dir, filename))
    #     processes.append(p1)
    #     p1.start()
                
            # get_label_and_image(project_dir, filename)

    # get_label_and_image(project_dir, "2020-02-05_14.20.33--NAs--T1354-GFP_Burst")
    
    # for p in processes: # Wait for end of all processes
    #     p.join()
            

    print("------------------ DONE ------------------")


def get_label_and_image(project_dir, image_files_queue: Queue):
        
    while image_files_queue.qsize() > 0:
        filename = image_files_queue.get()
        print("Started with", filename)
        label_and_image_YOLOv8(project_dir, filename)

def label_and_image_YOLOv8(project_dir, filename):
    image_path = str(project_dir) +"\\"+ f"ALL_IMS\\{filename}.ims"
    image_gen = fg.bioformatsGen(Path(image_path))
    # raise Exception("asd")
    # dataframe with analysis results from .ims file
    image_analysed_results_df = pd.read_hdf(Path(str(project_dir) +"/"+ f"ALL_FOURIER_h5/{filename}.h5"), mode="r", key="fourier")

    data_analysis_to_save = {
        "filename":[],	
        "frame":[],	
        "granule_id":[],
        "width":[],
        "height":[],	
        "filename_full":[]
    }

    process_bar = tqdm.tqdm(enumerate(image_gen))
    # process_bar = tqdm.tqdm(enumerate(range(1000)))
    for frame_num, frame in process_bar:
        # Update the progress bar to account for the number of frames
        if frame_num == 0:
            total_frames = frame_num
            process_bar.reset(total_frames)
    # for frame in image_gen:
        frame_id = frame_num
        valid_granule_fourier = image_analysed_results_df[(image_analysed_results_df['valid'] == True) & (image_analysed_results_df['frame'] == frame_id)]
        valid_granule_ids = valid_granule_fourier['granule_id'].unique()
        image_data = frame.im_data

        for valid_granule_id in valid_granule_ids: # For each valid granule in frame
        # for valid_granule_id in [5]: # For each valid granule in frame
            #  ------------------- Get boundry ------------------- 
            granule_fourier = valid_granule_fourier[(valid_granule_fourier['granule_id'] == valid_granule_id) & (valid_granule_fourier['frame'] == frame_id)]
            # granule_fourier_old = image_analysed_results_df[(image_analysed_results_df['granule_id'] == valid_granule_id) & (image_analysed_results_df['frame'] == frame_id)]
            # assert all(granule_fourier['granule_id'].compare(granule_fourier_old['granule_id'])), "Not the same"
            # assert all(granule_fourier['granule_id'] == granule_fourier_old['granule_id']), "Not the same"
            
            #  ------------------- Get image of granule ------------------- 
            bbox_left = granule_fourier['bbox_left'].iloc[0]
            bbox_right = granule_fourier['bbox_right'].iloc[0]
            bbox_top = granule_fourier['bbox_top'].iloc[0]
            bbox_bottom = granule_fourier['bbox_bottom'].iloc[0]
            granule_cutout_image = image_data[bbox_left:bbox_right, bbox_bottom:bbox_top]

            # if valid_granule_id == 1: #use 3,  2 is also good for testing
            if valid_granule_id == 1: #use 3,  2 is also good for testing
                plot_single(image_data, granule_cutout_image, valid_granule_id, granule_fourier)
                exit()
            # ------------------- Scaling the granule cutout ------------------- 
            original_image = Image.fromarray(granule_cutout_image)
            cutout_height, cutout_width = abs(bbox_left-bbox_right), abs(bbox_bottom - bbox_top)
            upscaled_image, xs_upscaled, ys_upscaled = scale_padding(original_image, (cutout_height, cutout_width), granule_fourier, NEW_MAX_HEIGHT = 1024, NEW_MAX_WIDTH = 1024)
            # upscaled_image, xs_upscaled, ys_upscaled = scale_padding((cutout_height, cutout_width), granule_fourier, NEW_MAX_HEIGHT = 1024, NEW_MAX_WIDTH = 1024)
            
            # ------------------- Create directional gradients -------------------
            crop_width, crop_height = granule_cutout_image.shape
            pr = _BoundaryExtractionGradient()
            x_centre = granule_fourier['x'].iloc[0]
            y_centre = granule_fourier['y'].iloc[0]
            x_pos_relative = x_centre - bbox_left
            y_pos_relative = y_centre - bbox_bottom 
            gradient_image = pr.process_image(granule_cutout_image, (x_pos_relative, y_pos_relative)) # Needs the relative centre
            # upscaled_gradient_image, _, _ = scale_padding(Image.fromarray(gradient_image), (cutout_height, cutout_width), granule_fourier, NEW_MAX_HEIGHT = 1024, NEW_MAX_WIDTH = 1024)
            # Turn from float32 into uint16
            # upscaled_gradient_image = ((upscaled_gradient_image * (2**16)) / np.max(upscaled_gradient_image)).astype('uint16')
            # upscaled_gradient_image = (upscaled_gradient_image / np.max(upscaled_gradient_image))
            gradient_image += np.abs(np.min(gradient_image))
            gradient_image = ((gradient_image * (2**16 - 1)) / np.max(gradient_image)).astype('uint16')
            upscaled_gradient_image, _, _ = scale_padding(Image.fromarray(gradient_image), (cutout_height, cutout_width), granule_fourier, NEW_MAX_HEIGHT = 1024, NEW_MAX_WIDTH = 1024)
            # upscaled_gradient_image = (upscaled_gradient_image / np.max(upscaled_gradient_image))
            assert np.min(upscaled_gradient_image) >= 0, f"{np.min(upscaled_gradient_image)}"
            # --------------------------------------------------------------------

            xs_pixels, ys_pixels = pixels_between_points(xs_upscaled, ys_upscaled)
            assert len(xs_pixels) == len(ys_pixels), f"They should have equal length {len(xs_pixels)} == {len(ys_pixels)}"

            # im_path = granule_fourier['im_path'].iloc[0]
            # im_path = Path(im_path).stem
            # if im_path == "2020-02-05_14.35.36--NAs--T1354-GFP_Burst":
            # plot(granule_cutout_image, upscaled_gradient_image, valid_granule_id, xs_upscaled, ys_upscaled, granule_fourier)
            # print(upscaled_gradient_image.dtype)
            # print(upscaled_image.dtype)
            # exit()
            # if valid_granule_id == 3: # 2 is also good for testing
            #     exit()

            # ------------------- Save label to .txt ------------------- 
            assert not granule_fourier.empty, "No fourier terms for valid granule. This should not be possible."
            # Create string, x0,y0,x1,y1,...,xn,yn 
            # This is the YOLOv8 segmentation mask format
            # zipped_normalized = list(zip(np.round(xs_pixels / 1024,6), np.round(ys_pixels / 1024, 6)))
            # coord_string = ''.join(map(lambda xy: str(xy[0]) + " " + str(xy[1]) + " ", zipped_normalized))
            # yolov8_granule_string = "0 " + coord_string + "\n" # 0 is the id of the class belonging to the mask created by the coords_string.
            # origin_ims_file = Path(frame.im_path).stem
            # with open(f"datasets/cutout_with_padding/all_data/labels/{origin_ims_file}_Frame_{frame.frame_num}_Granule_{valid_granule_id}.txt", "w+") as f:
            #     f.write(''.join(yolov8_granule_string))
            #     f.close()
            # ------------------- Save granule_cutout to .png ------------------- 
            # plt.imsave(f"datasets/cutout_with_padding/all_data/images/{filename}_Frame_{frame_num}_Granule_{valid_granule_id}.png.png", upscaled_image)
            assert upscaled_image.shape == (1024,1024), f"Wrong shape, was {upscaled_image.shape} should be (1024,1024)"
            # THIS ONE FOR ACTUAL GRANULE IMAGE
            # cv2.imwrite(f"D:/Master/MasterProject/dataset_creation/datasets/FINAL_DATASET_cutout_with_padding/images_grayscale_16bit/{filename}_Frame_{frame_num}_Granule_{valid_granule_id}.png", upscaled_image)
            # THIS ONE FOR GRADIENT IMAGE
            # cv2.imwrite(f"D:/Master/MasterProject/dataset_creation/datasets/FINAL_DATASET_cutout_with_padding/images_grayscale_16bit_gradient/{filename}_Frame_{frame_num}_Granule_{valid_granule_id}_gradient.png", upscaled_gradient_image)
            ### cv2.imwrite(f"datasets/cutout_with_padding/all_data/images_grayscale_16bit/{filename}_Frame_{frame_num}_Granule_{valid_granule_id}.png", upscaled_image)
            
            # ----------------- Save 2-channel image of granule and gradient -------------
            # image = np.expand_dims(upscaled_image, axis=-1)
            # gradient = np.expand_dims(upscaled_gradient_image, axis=-1)
            # upscaled_2channel_image = np.concatenate((image,gradient), axis=2)
            # np.save(f"D:/Master/MasterProject/dataset_creation/datasets/FINAL_DATASET_cutout_with_padding/images_grayscale_16bit_2channel/{filename}_Frame_{frame_num}_Granule_{valid_granule_id}_twochannel", upscaled_2channel_image)
            
            # ------------------- Save label as image (.png) ------------------- 
            # border_image = np.zeros((1024,1024))
            # for i in range(len(xs_pixels)):
            #     border_image[ys_pixels[i], xs_pixels[i]] = 1
            # flood_fill = ski.morphology.flood(border_image, (512,512), connectivity=1)
            # border_image[flood_fill == True] = 1
            ## plt.imsave(f"datasets/cutout_with_padding/all_data/labels_as_images/{filename}_Frame_{frame_num}_Granule_{valid_granule_id}.png", border_image)
            # THIS ONE FOR ACTUAL LABEL IMAGE
            # cv2.imwrite(f"D:/Master/MasterProject/dataset_creation/datasets/FINAL_DATASET_cutout_with_padding/labels_as_images/{filename}_Frame_{frame_num}_Granule_{valid_granule_id}.png", border_image) # TODO: Uncomment this
            # Might speed up saving: https://stackoverflow.com/questions/58231354/slow-matplotlib-savefig-to-png

            # Save info
            # data_analysis_to_save["filename"].append(filename)
            # data_analysis_to_save["frame"].append(frame_num)
            # data_analysis_to_save["granule_id"].append(valid_granule_id)
            # data_analysis_to_save["width"].append(bbox_right-bbox_left )
            # data_analysis_to_save["height"].append(bbox_top-bbox_bottom)
            # data_analysis_to_save["filename_full"].append(f"{filename}_Frame_{frame_num}_Granule_{valid_granule_id}.png")

            # plot(granule_cutout_image, upscaled_image, valid_granule_id, xs_upscaled, ys_upscaled, granule_fourier)
            # return
            # if frame_id == 1:
            #     return

    # data_analysis_to_save = pd.DataFrame(data_analysis_to_save)
    # data_analysis_to_save.to_csv(f"D:/Master/MasterProject/dataset_creation/datasets/FINAL_DATASET_cutout_with_padding/images_saved_analysis_data/{filename}.csv", index_label=False)
        
    # Clean up
    del image_analysed_results_df
    image_gen.close()
    del image_gen


def split_data_train_val(training_ratio=10):
    """ Split images and labels from source folders into training and validation folders """
    data_root = "datasets/cutout_with_padding/"
    label_path = data_root + "all_data/labels/" # Take files from here
    image_path = data_root + "all_data/images/" # Take files from here

    # Get a list of all files in the folder
    img_files = os.listdir(image_path)
    label_files = os.listdir(label_path)

    # 1000 frames, grab 100
    img_files_train, img_files_val = split_list_by_nth(img_files, training_ratio)
    label_files_train, label_files_val = split_list_by_nth(label_files, training_ratio)

    print("------------------ MOVING TRAINING DATA ------------------")
    # ------------------- Move training images & labels -------------------
    for i in range(len(img_files_train)):
        old_path_img = os.path.join(image_path, img_files_train[i])
        new_path_img = os.path.join(data_root+"images/train", img_files_train[i])    # Put files here
        assert img_files_train[i][:-4] == label_files_train[i][:-4], f"{i} - " + img_files_train[i][:-4] + "==" + label_files_train[i][:-4]
        shutil.copy(old_path_img, new_path_img)

        old_path_label = os.path.join(label_path, label_files_train[i])
        new_path_label = os.path.join(data_root+"labels/train", label_files_train[i]) # Put files here
        shutil.copy(old_path_label, new_path_label)
        # print(f'Copied: {old_path} -> {new_path}')
    
    print("------------------ MOVING VAL DATA ------------------")
    # ------------------- Move val images & labels -------------------
    for i in range(len(img_files_val)):
        old_path_img = os.path.join(image_path, img_files_val[i])
        new_path_img = os.path.join(data_root+"images/val", img_files_val[i])     # Put files here
        assert img_files_val[i][:-4] == label_files_val[i][:-4], f"{i} - " + img_files_val[i][:-4] + "==" + label_files_val[i][:-4]
        shutil.copy(old_path_img, new_path_img)

        old_path_label = os.path.join(label_path, label_files_val[i])
        new_path_label = os.path.join(data_root+"labels/val", label_files_val[i]) # Put files here
        shutil.copy(old_path_label, new_path_label)
        # print(f'Copied: {old_path} -> {new_path}')

    print("------------------ DONE ------------------")

def split_list_by_nth(list_to_split: list, split_number=10):
    """
        Split a list in two by some ratio, ie 1 to 10 ratio. Elements are split/taken every nth index.
        Used to split list into training and validation sets.
    """
    files_train = []
    files_val = []
    for i in range(len(list_to_split)):
        if i % split_number == 0:
            files_val.append(list_to_split[i])
        else:
            files_train.append(list_to_split[i])
    return files_train, files_val


def scale_stretch(original_image: Image, NEW_MAX_HEIGHT=1024, NEW_MAX_WIDTH=1024) -> np.array:
    """Stretches the image to new given size, irrespective of original aspect ratio.

    Args:
        original_image (Image): Image to stretch
        NEW_MAX_HEIGHT (int, optional): _description_. Defaults to 1024.
        NEW_MAX_WIDTH (int, optional): _description_. Defaults to 1024.

    Returns:
        np.array: Resized image
    """
    return np.array(original_image.resize((NEW_MAX_HEIGHT, NEW_MAX_WIDTH), resample=Image.Resampling.NEAREST))

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
    xs_upscaled = xs * scale_factor + scale_factor / 2 - 1/2 
    ys_upscaled = ys * scale_factor + scale_factor / 2 - 1/2
    # --- Add padding to border points ---
    xs_upscaled += delta_w // 2
    ys_upscaled += delta_h // 2
    upscaled_width, upscaled_height = new_im.size
    assert (upscaled_width, upscaled_height) == (NEW_MAX_WIDTH, NEW_MAX_HEIGHT), f"Should be {(NEW_MAX_WIDTH, NEW_MAX_HEIGHT)} == {(upscaled_width, upscaled_height)}"
    
    # return None, xs_upscaled, ys_upscaled
    return np.array(new_im), xs_upscaled, ys_upscaled




if __name__ == "__main__":
    fg.startVM()

    @fg.vmManager
    def main():
        generate_granule_cutout_images()
        # label_and_image_YOLOv8(project_dir="D:\Master\MasterProject\dataset_creation\data", filename="2020-02-05_14.18.49--NAs--T1354-GFP_Burst")
        # split_data_train_val(training_ratio=10) # TODO: Fix this argument train/val percentage
        # pass

    main()























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


# def scale_padding(original_image, img_dims: tuple[int,int], granule_fourier: pd.DataFrame, NEW_MAX_HEIGHT=1024, NEW_MAX_WIDTH=1024) -> tuple[np.array, np.array, np.array]:
#     # ------------------- Upscale image -------------------
#     cutout_height, cutout_width = img_dims
#     max_scale_height = int(np.floor(NEW_MAX_HEIGHT / cutout_height))
#     max_scale_width  = int(np.floor(NEW_MAX_WIDTH / cutout_width))
#     scale_factor = min(max_scale_height, max_scale_width) # Max amount to scale by while keeping aspect ratio
#     upscaled_image = original_image.resize((cutout_width*scale_factor, cutout_height*scale_factor), resample=Image.Resampling.NEAREST)
#     # ------------------- Add padding -------------------
#     # assert upscaled_image.size == (NEW_MAX_HEIGHT, NEW_MAX_WIDTH), f"New size of image is wrong. What? Was {upscaled_image.size} should be {NEW_MAX_HEIGHT, NEW_MAX_WIDTH}"
#     image_width, image_height = (cutout_width*scale_factor, cutout_height*scale_factor)
#     delta_w = NEW_MAX_WIDTH - image_width
#     delta_h = NEW_MAX_HEIGHT - image_height
#     padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
#     new_im = ImageOps.expand(upscaled_image, padding)
#     # ------------------- Get pixel border -------------------
#     xs, ys = get_coords(granule_fourier, get_relative=True)
#     xs = np.append(xs,xs[0]) # Add connection from last element to start element # TODO: Error is in here somewhere. Image upscaling is correct, problem with border? Titlted?
#     ys = np.append(ys,ys[0])
#     # --- Scale border points ---
#     xs_upscaled = xs * scale_factor + scale_factor / 2 - 1/2 
#     ys_upscaled = ys * scale_factor + scale_factor / 2 - 1/2
#     # --- Add padding to border points ---
#     xs_upscaled += delta_w // 2
#     ys_upscaled += delta_h // 2
#     upscaled_width, upscaled_height = new_im.size
#     assert (upscaled_width, upscaled_height) == (NEW_MAX_WIDTH, NEW_MAX_HEIGHT), f"Should be {(NEW_MAX_WIDTH, NEW_MAX_HEIGHT)} == {(upscaled_width, upscaled_height)}"
    
#     # return None, xs_upscaled, ys_upscaled
#     return np.array(new_im), xs_upscaled, ys_upscaled