import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7' # Restrict program to only see spesified gpu instance. Useful for multi-gpu machines.

import torch # Import torch after os env is set
ML_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('/Home/siv32/eto033/granule_explorer_core/experiments/ML_2019-10-31__1/best_model__DeepLabV3Plus__timm-efficientnet-b2__JaccardLoss__Freeze_encoder_False__two_channel__LR_0.001.pth')

import h5py
import pandas as pd
from shapely.geometry import Polygon
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
# from IPython.display import Image
from PIL import Image, ImageOps
import skimage as ski
import seaborn as sns 

from helper_functions.helper_functions import get_coords, pixels_between_points, _BoundaryExtractionGradient, scale_image_add_padding, scale_padding
# from dataset_creation.helper_functions.helper_functions import get_coords, pixels_between_points
from helper_functions import frame_gen as fg
from helper_functions.frame_gen import startVM, vmManager

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoLocator
import matplotlib.colors as mcol
font = {'family' : 'serif',
         'size'   : 26,
         'serif':  'cmr10'
         }
plt.rc('font', **font)
plt.rc('axes', unicode_minus=False)
plt.rcParams.update({'font.size': 26})
pd.options.mode.chained_assignment = None

from pathlib import Path
import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# helper function for data visualization
def visualize(**images, ):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        
    plt.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/debug_image{np.random.randint(0,1000000)}.png")
    # plt.show()

def visualize_better(granule_name, image,gradient,gradient_method, ML_method,border_float, boder_pixel):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(14, 10), layout='constrained')
    # Col 1
    a1 = sns.heatmap(image,ax=ax1, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
    a1.plot(border_float[0], border_float[1])
    a1.axis('off')
    a1.set_title("Base")

    # a4 = sns.heatmap(gradient,ax=ax4, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
    # a4.axis('off')
    # a4.set_title("Gradient")
    a4 = sns.heatmap(image,ax=ax4, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
    a4.plot(border_float[0], border_float[1])
    a4.axis('off')
    a4.set_title("Base")

    # Col 2
    a2 = sns.heatmap(gradient_method, ax=ax2, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
    a2.axis('off')
    a2.set_title("Gradient Method")

    a5 = sns.heatmap(ML_method, ax=ax5, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
    a5.axis('off')
    a5.set_title("ML Method")
    # Col 3
    a3 = sns.heatmap(gradient_method, ax=ax3, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
    a3.axis('off')
    a3.set_title("Overlap")

    a6 = sns.heatmap(image, ax=ax6, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
    sns.heatmap(ML_method, ax=ax6, rasterized=True, cmap=plt.colormaps["inferno"], cbar=False, alpha=0.5) 
    # ax6.imshow(ML_method, cmap=plt.colormaps["inferno"], alpha=0.5)
    a6.axis('off')
    a6.set_title("Overlap")

    fig.tight_layout()
    fig.suptitle(granule_name)
    fig.subplots_adjust(top=0.88)
    fig.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs2/debug_image{np.random.randint(0,1000000)}.png")

def ML_run_model(cropped_image, local_centre):
    original_image = Image.fromarray(cropped_image) 
    upscaled_image: np.array = scale_image_add_padding(original_image) 

    # Get and upscale granule gradients
    gradient_processor = _BoundaryExtractionGradient()
    gradient_granule_image: np.array = gradient_processor.process_image(cropped_image, local_centre)
    upscaled_gradient_granule_image: np.array = scale_image_add_padding(Image.fromarray(gradient_granule_image))
    # Scale gradient
    upscaled_gradient_granule_image += np.abs(np.min(upscaled_gradient_granule_image))
    upscaled_gradient_granule_image = ((upscaled_gradient_granule_image * (2**16 - 1)) / np.max(upscaled_gradient_granule_image)).astype('uint16')
    
    # Make correct dimentions for the model
    upscaled_image: np.array = np.expand_dims(upscaled_image, axis=0)
    upscaled_gradient_granule_image: np.array = np.expand_dims(upscaled_gradient_granule_image, axis=0)
    # Layer both images ontop of each other as a 2-channel image
    upscaled_2channel_image: np.array = np.concatenate((upscaled_image, upscaled_gradient_granule_image), axis=0).astype('float32') # Dims (channels, H, W) -> (2, H, W)


    upscaled_2channel_image: torch.Tensor = torch.from_numpy(upscaled_2channel_image).unsqueeze(0) # Dims (Batch size, channels, H, W) -> (1, 2, H, W)
    # Get model from globel context and make prediction
    model.to(ML_DEVICE)
    result_mask = model(upscaled_2channel_image.to(ML_DEVICE)) # Send image to compute device and make prediction

    ### result_mask = model(torch.from_numpy(upscaled_2channel_image).float().to("cuda").unsqueeze(0).unsqueeze(0))  # predict on an image
    ### result_mask = model(torch.from_numpy(np.array([upscaled_image2,upscaled_image2,upscaled_image2])).float().to("cuda").unsqueeze(0))  # predict on an image
    ### result_mask = model(rgb_array)  # predict on an image
    
    result_mask = result_mask.squeeze().detach().cpu().numpy().round()
    return result_mask

def read_csv(base_filename):
    "Reads in csv with what granules to plot, their origin file, frame id, granule id."
    base_csv = pd.read_csv(base_filename, index_col=0)
    return base_csv.reset_index(drop=True)
    # comp_df['file'] = pathlib.Path(full_path).stem 

def print_granules():
    csv = read_csv("test.csv")

    for i, file in enumerate(csv.groupby('file')):
        filename = file[0]
        data = file[1]
        print(f"\n Loading {filename} \n")

        try:
            # im_directory = "D:\Granule_experiment_data\ALL_IMS\ALL_IMS_TOGETHER"
            im_directory = f"/Home/siv32/eto033/ims_files/{filename[:10]}"
            ims_folders = [d for d in os.listdir(im_directory) if d[:4] == "data"]
            print(ims_folders)

            for root, dirs, files in os.walk(im_directory):
                if f"{filename}.ims" in files:
                    found_file = os.path.join(root, f"{filename}.ims")
                    # print(found_file)
                    break

            image_gen = fg.bioformatsGen(Path(found_file))
            next(image_gen)
            image_gen = fg.bioformatsGen(Path(found_file)) # Ensure .ims file actually exists, bioformats will not throw error untill it is used by next()...

            image_analysed_results_df = pd.read_hdf(Path(f"/Home/siv32/eto033/granule_explorer_core/experiments/ML_{filename[:10]}__1/fourier/{filename}--DEBUG.h5"), mode="r", key="fourier")
        except Exception as e:
            print("\nERROR\n")
            print(e)
            break

        print("\n\n")
        print(data.columns)
        print(data[:3])
        print("\n\n")
        # data_sorted = data[1].sort_values('frames')
        for i, image_data in enumerate(image_gen):
            for row in data[data['frames'] == i].iterrows(): # For all granules in a frame, grab image data
                print(row[1])
                wanted_frame = row[1]['frames']
                wanted_granule_id = row[1]['granule_ids']
                if i == wanted_frame:
                    print(f"Frame {wanted_frame}")
                    
                # ----------- Making figures ----------- 
                granule_fourier = image_analysed_results_df[(image_analysed_results_df['granule_id'] == wanted_granule_id) & (image_analysed_results_df['frame'] == wanted_frame)]
                assert granule_fourier['valid'].iloc[0] == True, f"Wanted granule with ID {wanted_granule_id} Frame {wanted_frame} is not valid!"

                bbox_left = granule_fourier['bbox_left'].iloc[0]
                bbox_right = granule_fourier['bbox_right'].iloc[0]
                bbox_top = granule_fourier['bbox_top'].iloc[0]
                bbox_bottom = granule_fourier['bbox_bottom'].iloc[0]
                granule_cutout_image = image_data.im_data[bbox_left:bbox_right, bbox_bottom:bbox_top]
                
                gradient_processor = _BoundaryExtractionGradient()
                x_centre = granule_fourier['x'].iloc[0]
                y_centre = granule_fourier['y'].iloc[0]
                x_pos_relative = x_centre - bbox_left
                y_pos_relative = y_centre - bbox_bottom 
                upscaled_gradient_granule_image = gradient_processor.process_image(granule_cutout_image, (x_pos_relative, y_pos_relative)) 

                try:
                    # ------------ Pixel ground truth ------------ 
                    original_image = Image.fromarray(granule_cutout_image)
                    cutout_height, cutout_width = abs(bbox_left-bbox_right), abs(bbox_bottom - bbox_top)
                    _, xs_upscaled, ys_upscaled = scale_padding(original_image, (cutout_height, cutout_width), granule_fourier, NEW_MAX_HEIGHT = 1024, NEW_MAX_WIDTH = 1024)
                    assert (np.max(xs_upscaled) <= 1024) and (np.max(ys_upscaled) <= 1024), f"Number out of bounds \n x_ints: {np.max(xs_upscaled)} y_ints: {np.max(ys_upscaled)})"
                    
                    xs_pixels, ys_pixels = pixels_between_points(xs_upscaled, ys_upscaled)
                    assert len(xs_pixels) == len(ys_pixels), f"They should have equal length {len(xs_pixels)} == {len(ys_pixels)}"

                    border_image = np.zeros((1024,1024))
                    print("\n")
                    print("Length", np.max(xs_pixels))
                    print("\n")
                    for i in range(len(xs_pixels)):
                        border_image[ys_pixels[i], xs_pixels[i]] = 1
                    flood_fill = ski.morphology.flood(border_image, (512,512), connectivity=1)
                    border_image[flood_fill == True] = 1
                except:
                    continue
                # ------------ ML Output ------------ 
                result_mask = ML_run_model(granule_cutout_image, (x_pos_relative, y_pos_relative))

                # visualize(
                #     image=scale_image_add_padding(Image.fromarray(granule_cutout_image)),
                #     gradient=scale_image_add_padding(Image.fromarray(upscaled_gradient_granule_image)),
                #     gradient_method=border_image, 
                #     ML_method=result_mask
                # )   

                visualize_better(
                    granule_name=f"{filename[:19]} F: {wanted_frame} ID: {wanted_granule_id}",
                    image=scale_image_add_padding(Image.fromarray(granule_cutout_image)),
                    gradient=scale_image_add_padding(Image.fromarray(upscaled_gradient_granule_image)),
                    gradient_method=border_image, 
                    ML_method=result_mask,
                    border_float=(xs_upscaled, ys_upscaled),
                    boder_pixel=(xs_pixels, ys_pixels)
                )   

        #         return
        # break

    # for i, im in enumerate(image_gen):
    #     print(i, im)
    #     break


if __name__ == "__main__":
    fg.startVM()

    @fg.vmManager
    def main():
        print_granules()

    main()







