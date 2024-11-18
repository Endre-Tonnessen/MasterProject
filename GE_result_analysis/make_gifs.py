import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6' # Restrict program to only see spesified gpu instance. Useful for multi-gpu machines.

import torch # Import torch after os env is set
ML_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = torch.load('/Home/siv32/eto033/granule_explorer_core/experiments/best_model__DeepLabV3Plus__timm-efficientnet-b2__JaccardLoss__Freeze_encoder_False__two_channel__LR_0.001.pth', map_location=torch.device(ML_DEVICE))
MODEL.to(ML_DEVICE)
MODEL.eval()

import tqdm
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
import cv2
import imageio
from helper_functions.helper_functions import get_coords, pixels_between_points, _BoundaryExtractionGradient, scale_image_add_padding, scale_padding
from helper_functions import frame_gen as fg
from helper_functions.frame_gen import startVM, vmManager
from matplotlib import cm

import matplotlib as mplt
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
"""
For all frames
    For all granules in frame
        Add granule cutout to dict

        Add granule gradient to dict
        If gradient valid, calc shape, add to dict
        If ML valid, calc shaape, add to dict

"""

def convert_(img):
    my_cm = mplt.cm.get_cmap('viridis')
    normed_data = (img - np.min(img)) / (np.max(img) - np.min(img))
    mapped_data = my_cm(normed_data)
    mapped_datau8 = (255 * my_cm(normed_data)).astype('uint8')
    return mapped_datau8


def make_gifs():
    filename_ = "2020-02-05_14.18.49--NAs--T1354-GFP_Burst"

    if not os.path.exists(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/{filename_}"):
        os.makedirs(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/{filename_}")
    if not os.path.exists(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/{filename_}/imgs"):
        os.makedirs(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/{filename_}/imgs")

    image_gen = fg.bioformatsGen(Path(f"/Home/siv32/eto033/ims_files/2020-02-05/{filename_}.ims")) # Generator for frames
    image_analysed_results_df_gradient = pd.read_hdf(Path(f"/Home/siv32/eto033/fourier_files/gradient/2020-02-05/fourier/{filename_}--DEBUG.h5"), mode="r", key="fourier")
    image_analysed_results_df_ML = pd.read_hdf(Path(f"/Home/siv32/eto033/fourier_files/ML/2020-02-05/fourier/{filename_}--DEBUG.h5"), mode="r", key="fourier")

    """
    granule_id: [
        frame: {
            base_image: {img}
            gradient_image: {img}
            ML: {
                mask_img, if_valid 
            }
            Gradient: {
                mask_img, if_valid 
            }
        }
    ]
    """
    granules = {}

    process_bar = tqdm.tqdm(enumerate(image_gen))
    # process_bar = tqdm.tqdm(enumerate(range(1000)))
    for frame_num, frame in process_bar:
        # Update the progress bar to account for the number of frames
        if frame_num == 0:
            total_frames = frame_num
            process_bar.reset(total_frames)
        # if frame_num == 500:
        #     break
        
        frame_id = frame_num
        valid_granule_fourier = image_analysed_results_df_gradient[(image_analysed_results_df_gradient['frame'] == frame_id)]                                                  # For ALL granules in frame
        # valid_granule_fourier = image_analysed_results_df_gradient[(image_analysed_results_df_gradient['valid'] == True) & (image_analysed_results_df_gradient['frame'] == frame_id)] # For ALL VALID granules in frame 
        granule_ids = valid_granule_fourier['granule_id'].unique()
        image_data = frame.im_data

        for granule_id in granule_ids: # For each valid granule in frame
            granule_fourier = valid_granule_fourier[(valid_granule_fourier['granule_id'] == granule_id) & (valid_granule_fourier['frame'] == frame_id)]

            #  ------------------- Get image of granule ------------------- 
            bbox_left = granule_fourier['bbox_left'].iloc[0]
            bbox_right = granule_fourier['bbox_right'].iloc[0]
            bbox_top = granule_fourier['bbox_top'].iloc[0]
            bbox_bottom = granule_fourier['bbox_bottom'].iloc[0]
            granule_cutout_image = image_data[bbox_left:bbox_right, bbox_bottom:bbox_top]
            # ------------------- Scaling the granule cutout ------------------- 
            original_image = Image.fromarray(granule_cutout_image)
            cutout_height, cutout_width = abs(bbox_left-bbox_right), abs(bbox_bottom - bbox_top)
            upscaled_image, xs_upscaled, ys_upscaled = scale_padding(original_image, (cutout_height, cutout_width), granule_fourier, NEW_MAX_HEIGHT = 1024, NEW_MAX_WIDTH = 1024)
            
            # ------------------- Create directional gradients -------------------
            crop_width, crop_height = granule_cutout_image.shape
            pr = _BoundaryExtractionGradient()
            x_centre = granule_fourier['x'].iloc[0]
            y_centre = granule_fourier['y'].iloc[0]
            x_pos_relative = x_centre - bbox_left
            y_pos_relative = y_centre - bbox_bottom 
            gradient_image = pr.process_image(granule_cutout_image, (x_pos_relative, y_pos_relative)) # Needs the relative centre

            gradient_image += np.abs(np.min(gradient_image))
            gradient_image = ((gradient_image * (2**16 - 1)) / np.max(gradient_image)).astype('uint16')
            upscaled_gradient_image, _, _ = scale_padding(Image.fromarray(gradient_image), (cutout_height, cutout_width), granule_fourier, NEW_MAX_HEIGHT = 1024, NEW_MAX_WIDTH = 1024)

            # ------------------ ML ------------------
            if not image_analysed_results_df_ML[(image_analysed_results_df_ML['valid'] == True) 
                                    & (image_analysed_results_df_ML['frame'] == frame_id)
                                    & (image_analysed_results_df_ML['granule_id'] == granule_id)].empty:
                # Make correct dimentions for the model
                upscaled_image_ml: np.array = np.expand_dims(upscaled_image, axis=0) # (1, H, W)
                upscaled_gradient_granule_image: np.array = np.expand_dims(upscaled_gradient_image, axis=0) # (1, H, W)
                # Layer both images ontop of each other as a 2-channel image
                upscaled_2channel_image: np.array = np.concatenate((upscaled_image_ml, upscaled_gradient_granule_image), axis=0).astype('float32') # Dims (channels, H, W) -> (2, H, W)
                upscaled_2channel_image: torch.Tensor = torch.from_numpy(upscaled_2channel_image).unsqueeze(0) # Dims (Batch size, channels, H, W) -> (1, 2, H, W)

                result_mask = MODEL(upscaled_2channel_image.to(ML_DEVICE)) # Send image to compute device and make prediction
                result_mask = result_mask.squeeze().detach().cpu().numpy().round()
            else:
                result_mask = None
            # ----------------------------------------

            # ------------------ Gradient border ------------------
            if not image_analysed_results_df_gradient[(image_analysed_results_df_gradient['valid'] == True) 
                                                & (image_analysed_results_df_gradient['frame'] == frame_id)
                                                & (image_analysed_results_df_gradient['granule_id'] == granule_id)].empty:
                xs_pixels, ys_pixels = pixels_between_points(xs_upscaled, ys_upscaled)

                try: # In case of failure, ignore granule and move on.
                    border_image = np.zeros((1024,1024))
                    for i in range(len(xs_pixels)):
                        border_image[ys_pixels[i], xs_pixels[i]] = 1
                    flood_fill = ski.morphology.flood(border_image, (512,512), connectivity=1)
                    border_image[flood_fill == True] = 1
                except:
                    border_image = None
            else:
                border_image = None


            # Add to dict
            # granules.setdefault(str(granule_id), []).append(
            #     {
            #         "base_image": upscaled_image,
            #         "base_gradient_image": upscaled_gradient_image,
            #         "result_mask": result_mask,
            #         "border_image": border_image
            #     }
            # )
            
            cv2.imwrite(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/{filename_}/imgs/granule_{granule_id}_Frame_{frame_num}.png",          cv2.applyColorMap(((upscaled_image/np.max(upscaled_image)*255)).astype(np.uint8), cv2.COLORMAP_VIRIDIS))
            cv2.imwrite(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/{filename_}/imgs/granule_gradient_{granule_id}_Frame_{frame_num}.png", cv2.applyColorMap(((upscaled_gradient_image/np.max(upscaled_gradient_image)*255)).astype(np.uint8), cv2.COLORMAP_VIRIDIS))
            if result_mask is not None:
                cv2.imwrite(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/{filename_}/imgs/ML_mask{granule_id}_Frame_{frame_num}.png",       cv2.applyColorMap(((result_mask/np.max(result_mask)*255)).astype(np.uint8),       cv2.COLORMAP_VIRIDIS))
            if border_image is not None:
                cv2.imwrite(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/{filename_}/imgs/Gradient_mask{granule_id}_Frame_{frame_num}.png", cv2.applyColorMap(((border_image/np.max(border_image)*255)).astype(np.uint8),     cv2.COLORMAP_VIRIDIS))

    
    # for k in granules.keys():
    #     if not os.path.exists(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/{filename_}/{k}"):
    #         os.makedirs(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/{filename_}/{k}")

    #     granules_in_dict: list[dict] = granules[k] 

    #     # Save still image
    #     # plt.figure(frameon=False)
    #     # plt.imshow(granules_in_dict[0]["base_image"], cmap=plt.colormaps["viridis"])
    #     # plt.axis('off')
    #     # plt.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/test/asd.png", bbox_inches='tight', pad_inches=0)
    #     # plt.close()

    #     granule_base_img    = list(map(lambda g: convert_(g["base_image"]),          granules_in_dict))
    #     base_gradient_image = list(map(lambda g: convert_(g["base_gradient_image"]), granules_in_dict))

    #     result_mask         = list(map(lambda g: convert_(g["result_mask"]), 
    #                                 list(filter(g["result_mask"] != None, granules_in_dict))))
    #     border_image        = list(map(lambda g: convert_(g["border_image"]), 
    #                                 list(filter(g["border_image"] != None, granules_in_dict))))


    #     # Save gif
    #     imageio.mimsave(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/test/{filename_}/{k}/granule_{k}.gif", granule_base_img)
    #     imageio.mimsave(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/test/{filename_}/{k}/granule_gradient_{k}.gif", base_gradient_image)
    #     imageio.mimsave(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/test/{filename_}/{k}/ML_mask{k}.gif", result_mask)
    #     imageio.mimsave(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/gifs/test/{filename_}/{k}/Gradient_mask{k}.gif", border_image)




if __name__ == "__main__":
    fg.startVM()

    @fg.vmManager
    def main():
        make_gifs()
        print("----------- Done! -----------")

    main()