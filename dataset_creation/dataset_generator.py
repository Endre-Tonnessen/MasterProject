import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
import os
import shutil
import tqdm
import cv2

import bioformats as bf
import javabridge
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helper_functions.helper_functions import get_coords, pixels_between_points
# from dataset_creation.helper_functions.helper_functions import get_coords, pixels_between_points
from helper_functions import frame_gen as fg
# import dataset_creation.helper_functions.frame_gen as fg

import pathlib
import platform
plt2 = platform.system()
if plt2 == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

def plot(granule_cutout_image, upscaled_image, valid_granule_id, xs_upscaled, ys_upscaled, granule_fourier):
    fig = make_subplots(rows=1, cols=2, 
                    horizontal_spacing=0.05, 
                    vertical_spacing=0.1,
                    subplot_titles=('Standard image', 'Upscale Nearest'))
    xs, ys = get_coords(granule_fourier, get_relative=True)
    xs = np.append(xs,xs[0])
    ys = np.append(ys,ys[0])
    base_image_fig = px.imshow(granule_cutout_image)
    fig.add_trace(base_image_fig.data[0], 1, 1)                
    # Calculate and draw boundry for first plot
    xs_pixels, ys_pixels = pixels_between_points(xs, ys)
    fig.add_trace(go.Scatter(x=xs, y=ys, marker=dict(color='red', size=16), name=f"400 p border {valid_granule_id}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs_pixels, y=ys_pixels, marker=dict(color='cyan', size=16), name=f"Pixel border {valid_granule_id}"), row=1, col=1)

    upscaled_image_fig = px.imshow(upscaled_image)
    fig.add_trace(upscaled_image_fig.data[0], 1, 2)

    xs_pixels, ys_pixels = pixels_between_points(xs_upscaled, ys_upscaled)
    fig.add_trace(go.Scatter(x=xs_pixels, y=ys_pixels, marker=dict(color='cyan', size=16), name=f"Pixel border {valid_granule_id}"), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs_upscaled, y=ys_upscaled, marker=dict(color='red', size=16), name=f"400p upscaled border3 {valid_granule_id}"), row=1, col=2)
    
    fig.update_layout(title_text=f"Granule {valid_granule_id}", title_x=0.5, showlegend=False, font_size=11)
    fig.update_yaxes(autorange='reversed') # Ensure granules are not flipped. Plotly has strange axis direction defaults...
    fig.show()


def generate_granule_cutout_images(ims_file_directory_path: Path = "", 
                                   h5_analyzed_ims_data_path: Path = ""):
    """
    """
    current_file = Path(__file__).resolve() # TODO: FIX THIS PATH MESS
    project_dir = current_file.parents[1] / "dataset_creation"
    image_filename = "2020-02-05_15.41.32-NAs-T1354-GFP_Burst.ims"
    image_path = project_dir / f"data/{image_filename}"
    image_gen = fg.bioformatsGen(image_path)
    # dataframe with analysis results from .ims file
    image_analysed_results_df = pd.read_hdf(Path(project_dir / "data/Analysis_data/2020-02-05_15.41.32-NAs-T1354-GFP_Burst.h5"), mode="r", key="fourier")
   
    process_bar = tqdm.tqdm(enumerate(image_gen))
    for frame_num, frame in process_bar:
        # Update the progress bar to account for the number of frames
        if frame_num == 0:
            total_frames = frame.total_frames
            process_bar.reset(total_frames)

        frame_id = frame.frame_num
        valid_granule_fourier = image_analysed_results_df[(image_analysed_results_df['valid'] == True) & (image_analysed_results_df['frame'] == frame_id)]['granule_id']
        valid_granule_ids = valid_granule_fourier.unique()
        image_data = frame.im_data

        for valid_granule_id in valid_granule_ids: # For each valid granule in frame
        # for valid_granule_id in [5]: # For each valid granule in frame
            #  ------------------- Get boundry ------------------- 
            granule_fourier = image_analysed_results_df[(image_analysed_results_df['granule_id'] == valid_granule_id) & (image_analysed_results_df['frame'] == frame_id)]
            #  ------------------- Get image of granule ------------------- 
            bbox_left = granule_fourier['bbox_left'].iloc[0]
            bbox_right = granule_fourier['bbox_right'].iloc[0]
            bbox_top = granule_fourier['bbox_top'].iloc[0]
            bbox_bottom = granule_fourier['bbox_bottom'].iloc[0]
            granule_cutout_image = image_data[bbox_left:bbox_right, bbox_bottom:bbox_top]
            # ------------------- Scaling the granule cutout ------------------- 
            original_image = Image.fromarray(granule_cutout_image)
            upscaled_image, xs_upscaled, ys_upscaled = scale_padding(original_image, granule_fourier, NEW_MAX_HEIGHT = 1024, NEW_MAX_WIDTH = 1024)

            xs_pixels, ys_pixels = pixels_between_points(xs_upscaled, ys_upscaled)
            assert len(xs_pixels) == len(ys_pixels), f"They should have equal length {len(xs_pixels)} == {len(ys_pixels)}"

            # ------------------- Save label to .txt ------------------- 
            assert not granule_fourier.empty, "No fourier terms for valid granule. This should not be possible."
            # Create string, x0,y0,x1,y1,...,xn,yn 
            # This is the YOLOv8 segmentation mask format
            zipped_normalized = list(zip(np.round(xs_pixels / 1024,5), np.round(ys_pixels / 1024, 5)))
            coord_string = ''.join(map(lambda xy: str(xy[0]) + " " + str(xy[1]) + " ", zipped_normalized))
            yolov8_granule_string = "0 " + coord_string + "\n" # 0 is the id of the class belonging to the mask created by the coords_string.
            origin_ims_file = Path(frame.im_path).stem
            with open(f"datasets/cutout_dataset/all_data/labels/{origin_ims_file}_Frame_{frame.frame_num}_Granule_{valid_granule_id}.txt", "w+") as f:
                f.write(''.join(yolov8_granule_string))
                f.close()
            # ------------------- Save granule_cutout to .png ------------------- 
            plt.imsave(f"datasets/cutout_dataset/all_data/images/{origin_ims_file}_Frame_{frame.frame_num}_Granule_{valid_granule_id}.png", upscaled_image)

            # plot(granule_cutout_image, upscaled_image, valid_granule_id, xs_upscaled, ys_upscaled, granule_fourier)
            # if frame_id == 1:
            #     return


    print("------------------ DONE ------------------")



def split_data_train_val(training_ratio=10):
    """ Split images and labels from source folders into training and validation folders """
    data_root = "datasets/cutout_dataset/"
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

def scale_padding(original_image: Image, granule_fourier: pd.DataFrame, NEW_MAX_HEIGHT=1024, NEW_MAX_WIDTH=1024) -> tuple[np.array, np.array, np.array]:
    # ------------------- Upscale image -------------------
    cutout_height, cutout_width = np.array(original_image).shape[:2]
    max_scale_height = int(np.floor(NEW_MAX_HEIGHT / cutout_height))
    max_scale_width  = int(np.floor(NEW_MAX_WIDTH / cutout_width))
    scale_factor = min(max_scale_height, max_scale_width) # Max amount to scale by while keeping aspect ratio
    upscaled_image = original_image.resize((cutout_width*scale_factor, cutout_height*scale_factor), resample=Image.Resampling.NEAREST)
    # ------------------- Add padding -------------------
    image_width, image_height = upscaled_image.size
    delta_w = NEW_MAX_WIDTH - image_width
    delta_h = NEW_MAX_HEIGHT - image_height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(upscaled_image, padding)
    # ------------------- Get pixel border -------------------
    xs, ys = get_coords(granule_fourier, get_relative=True)
    xs = np.append(xs,xs[0]) # Add connection from last element to start element
    ys = np.append(ys,ys[0])
    # --- Scale border points ---
    xs_upscaled = xs * scale_factor + scale_factor / 2 - 1/2 
    ys_upscaled = ys * scale_factor + scale_factor / 2 - 1/2
    # --- Scale border points ---
    xs_upscaled += delta_w // 2
    ys_upscaled += delta_h // 2
    upscaled_width, upscaled_height = new_im.size
    assert (upscaled_width, upscaled_height) == (NEW_MAX_WIDTH, NEW_MAX_HEIGHT), f"Should be {(NEW_MAX_WIDTH, NEW_MAX_HEIGHT)} == {(upscaled_width, upscaled_height)}"
    
    return np.array(new_im), xs_upscaled, ys_upscaled

if __name__ == "__main__":
    fg.startVM()

    @fg.vmManager
    def main():
        # generate_granule_cutout_images()
        split_data_train_val(training_ratio=10) # TODO: Fix this argument train/val percentage
        # pass

    main()



    # coords_tuple = [(xs_pixels[i], ys_pixels[i]) for i in range(len(xs_pixels))]
    # seen = set()
    # coords_set = [x for x in coords_tuple if x not in seen and not seen.add(x)] # TODO: Might not be needed, pixels_between_points cannot produce duplicate coords anymore? 
    # xs = np.array([xy[0] for xy in coords_set])
    # ys = np.array([xy[1] for xy in coords_set])

    # scaling_dict = {'x_width_scale':[], 'y_height_scale':[], 'x_width_original_pixels':[], 'y_height_original_pixels':[], 'granule_id':[], 'frame_id':[], 'origin_filename':[]}
    # scaling_df: pd.DataFrame = pd.DataFrame(scaling_dict)
    # scaling_df.to_csv("scaling_df.csv")