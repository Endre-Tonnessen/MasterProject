import dataset_creation.helper_functions.frame_gen as fg
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import shutil

import bioformats as bf
import javabridge
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataset_creation.helper_functions.helper_functions import get_coords, pixels_between_points

import pathlib
import platform
plt2 = platform.system()
if plt2 == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

def plot(granule_cutout_image, upscaled_image, granule_fourier, valid_granule_id, scale_factor_width, scale_factor_height):
    fig = make_subplots(rows=1, cols=2, 
                    horizontal_spacing=0.05, 
                    vertical_spacing=0.1,
                    subplot_titles=('Standard image', 'Upscale Nearest'))
    
    base_image_fig = px.imshow(granule_cutout_image)
    fig.add_trace(base_image_fig.data[0], 1, 1)                
    # Calculate and draw boundry for first plot
    xs,ys = get_coords(granule_fourier, get_relative=True)
    xs2, ys2 = pixels_between_points(np.round(np.append(xs,xs[0]),0),np.round(np.append(ys,ys[0]),0), precision=10)
    fig.add_trace(go.Scatter(x=np.append(xs,xs[0]), y=np.append(ys,ys[0]), marker=dict(color='lightgreen', size=16), name=f"400 p border {valid_granule_id}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs2,y=ys2, marker=dict(color='cyan', size=16), name=f"True pixel border {valid_granule_id}"), row=1, col=1)

    upscaled_image_fig = px.imshow(upscaled_image)
    fig.add_trace(upscaled_image_fig.data[0], 1, 2)
    # Add borders for upscaled images
    xs3, ys3 = pixels_between_points(np.append(xs,xs[0]), np.append(ys,ys[0]), precision=100, scale_factor_x=scale_factor_width, scale_factor_y=scale_factor_height)
    coords_tuple = [(xs3[i], ys3[i]) for i in range(len(ys3))]
    # print("Length:", len(coords_))
    seen = set()
    coords_set = [x for x in coords_tuple if x not in seen and not seen.add(x)]
    xs3 = [xy[0] for xy in coords_set]
    ys3 = [xy[1] for xy in coords_set]
    # print("Length:", len(result))

    fig.add_trace(go.Scatter(x=np.append(xs,xs[0])*scale_factor_width, y=np.append(ys,ys[0])*scale_factor_height, marker=dict(color='lightgreen', size=16), name=f"100 p border {valid_granule_id}"), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs3, y=ys3, marker=dict(color='cyan', size=16), name=f"True pixel border {valid_granule_id}"), row=1, col=2)
    
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

    scaling_dict = {'x_width_scale':[], 'y_height_scale':[], 'x_width_original_pixels':[], 'y_height_original_pixels':[], 'granule_id':[], 'frame_id':[], 'origin_filename':[]}
   
    for frame in image_gen: 
        # frame: fg.MicroscopeFrame = next(image_gen)
        frame_id = frame.frame_num

        valid_granule_fourier = image_analysed_results_df[(image_analysed_results_df['valid'] == True) & (image_analysed_results_df['frame'] == frame_id)]['granule_id']
        valid_granule_ids = valid_granule_fourier.unique()

        for valid_granule_id in valid_granule_ids: # For each valid granule in frame
        # for valid_granule_id in [5]: # For each valid granule in frame
            #  ------------------- Get boundry ------------------- 
            granule_fourier = image_analysed_results_df[(image_analysed_results_df['granule_id'] == valid_granule_id) & (image_analysed_results_df['frame'] == frame_id)]
            #  ------------------- Get image of granule ------------------- 
            bbox_left = granule_fourier['bbox_left'].iloc[0]
            bbox_right = granule_fourier['bbox_right'].iloc[0]
            bbox_top = granule_fourier['bbox_top'].iloc[0]
            bbox_bottom = granule_fourier['bbox_bottom'].iloc[0]
            
            image_data = frame.im_data
            granule_cutout_image = image_data[bbox_left:bbox_right, bbox_bottom:bbox_top]
            # ------------------- Scaling the granule cutout ------------------- TODO: Make scaling preserve original cutout ratio
            NEW_MAX_WIDTH = 1024
            NEW_MAX_HEIGHT = 1024
            original_image = Image.fromarray(granule_cutout_image)
            original_width, original_height = original_image.size
            scale_factor_height = NEW_MAX_HEIGHT / original_height
            scale_factor_width = NEW_MAX_WIDTH / original_width
            assert scale_factor_height*original_height == NEW_MAX_HEIGHT, f"Should be {NEW_MAX_HEIGHT} was {scale_factor_height*original_height}"
            assert scale_factor_width*original_width == NEW_MAX_WIDTH, f"Should be {NEW_MAX_WIDTH} was {scale_factor_width*original_width}"
            # ------------------- Add scaling info to df ------------------- TODO: Might not need this
            scaling_dict['x_width_scale'].append(scale_factor_width)
            scaling_dict['y_height_scale'].append(scale_factor_height)
            scaling_dict['x_width_original_pixels'].append(original_width)
            scaling_dict['y_height_original_pixels'].append(original_height)
            scaling_dict['granule_id'].append(valid_granule_id)
            scaling_dict['frame_id'].append(frame_id)
            scaling_dict['origin_filename'].append(image_filename)
            # ------------------- Rescale image ------------------- 
            upscaled_image = np.array(original_image.resize((NEW_MAX_WIDTH, NEW_MAX_WIDTH), resample=Image.Resampling.NEAREST)) 
            # ------------------- Get pixel boundry -------------------  TODO: Create better `pixels_between_points()` that cannot miss pixels 
            xs,ys = get_coords(granule_fourier, get_relative=True)
            xs_pixels, ys_pixels = pixels_between_points(np.append(xs,xs[0]), np.append(ys,ys[0]), precision=100, scale_factor_x=scale_factor_width, scale_factor_y=scale_factor_height)
            coords_tuple = [(xs_pixels[i], ys_pixels[i]) for i in range(len(xs_pixels))]
            seen = set()
            coords_set = [x for x in coords_tuple if x not in seen and not seen.add(x)]
            xs = np.array([xy[0] for xy in coords_set])
            ys = np.array([xy[1] for xy in coords_set])
            # ------------------- Save label to .txt ------------------- 
            assert not granule_fourier.empty, "No fourier terms for valid granule. This should not be possible."
            # Create string, x0,y0,x1,y1,...,xn,yn 
            # This is the YOLOv8 segmentation mask format
            zipped_normalized = list(zip(np.round(xs / 1024,5), np.round(ys / 1024, 5)))
            coord_string = ''.join(map(lambda xy: str(xy[0]) + " " + str(xy[1]) + " ", zipped_normalized))
            yolov8_granule_string = "0 " + coord_string + "\n" # 0 is the id of the class belonging to the mask created by the coords_string.
            origin_ims_file = Path(frame.im_path).stem
            with open(f"datasets/cutout_dataset/all_data/labels/{origin_ims_file}_Frame_{frame.frame_num}_Granule_{valid_granule_id}.txt", "w+") as f:
                f.write(''.join(yolov8_granule_string))
                f.close()
            # ------------------- Save granule_cutout to .png ------------------- 
            plt.imsave(f"datasets/cutout_dataset/all_data/images/{origin_ims_file}_Frame_{frame.frame_num}_Granule_{valid_granule_id}.png", upscaled_image)


            # plot(granule_cutout_image, upscaled_image, granule_fourier, valid_granule_id, scale_factor_width, scale_factor_height)
        
        scaling_df: pd.DataFrame = pd.DataFrame(scaling_dict)
        scaling_df.to_csv("scaling_df.csv")

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

    # ------------------- Move training images & labels -------------------
    for i in range(len(img_files_train)):
        old_path_img = os.path.join(image_path, img_files_train[i])
        new_path_img = os.path.join(data_root+"train/images", img_files_train[i])    # Put files here
        assert img_files_train[i][:-4] == label_files_train[i][:-4], f"{i} - " + img_files_train[i][:-4] + "==" + label_files_train[i][:-4]
        shutil.copy(old_path_img, new_path_img)

        old_path_label = os.path.join(label_path, label_files_train[i])
        new_path_label = os.path.join(data_root+"train/labels", label_files_train[i]) # Put files here
        shutil.copy(old_path_label, new_path_label)
        # print(f'Copied: {old_path} -> {new_path}')
    
    # ------------------- Move val images & labels -------------------
    for i in range(len(img_files_val)):
        old_path_img = os.path.join(image_path, img_files_val[i])
        new_path_img = os.path.join(data_root+"val/images", img_files_val[i])     # Put files here
        assert img_files_val[i][:-4] == label_files_val[i][:-4], f"{i} - " + img_files_val[i][:-4] + "==" + label_files_val[i][:-4]
        shutil.copy(old_path_img, new_path_img)

        old_path_label = os.path.join(label_path, label_files_val[i])
        new_path_label = os.path.join(data_root+"val/labels", label_files_val[i]) # Put files here
        shutil.copy(old_path_label, new_path_label)
        # print(f'Copied: {old_path} -> {new_path}')


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

if __name__ == "__main__":
    fg.startVM()

    @fg.vmManager
    def main():
        # generate_granule_cutout_images()
        # split_data_train_val(training_ratio=10)
        pass

    main()


