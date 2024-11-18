import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6' # Restrict program to only see spesified gpu instance. Useful for multi-gpu machines.

import torch # Import torch after os env is set
ML_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('/Home/siv32/eto033/granule_explorer_core/experiments/best_model__DeepLabV3Plus__timm-efficientnet-b2__JaccardLoss__Freeze_encoder_False__two_channel__LR_0.001.pth')

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
    # fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), layout='constrained')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(14, 10), layout='constrained')
    # Col 1
    a1 = sns.heatmap(image,ax=ax1, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
    a1.plot(border_float[0], border_float[1], color='red')
    a1.axis('off')
    a1.set_title("Base")

    # a4 = sns.heatmap(gradient,ax=ax4, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
    # a4.axis('off')
    # a4.set_title("Gradient")
    a4 = sns.heatmap(image,ax=ax4, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
    a4.plot(border_float[0], border_float[1], color='red')
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
    a3.plot(border_float[0], border_float[1], color='red')
    a3.axis('off')
    a3.set_title("Overlap")

    a6 = sns.heatmap(ML_method, ax=ax6, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
    # sns.heatmap(ML_method, ax=ax6, rasterized=True, cmap=plt.colormaps["inferno"], cbar=False, alpha=0.5) 
    a6.plot(border_float[0], border_float[1], color='red')
    # ax6.imshow(ML_method, cmap=plt.colormaps["inferno"], alpha=0.5)
    a6.axis('off')
    a6.set_title("Overlap")

    fig.tight_layout()
    fig.suptitle(granule_name)
    fig.subplots_adjust(top=0.88)
    fig.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs3/debug_image{np.random.randint(0,1000000)}.png")


# def visualize_for_thesis_model_output(granule_name, image,gradient,gradient_method, ML_method,border_float, boder_pixel):
def visualize_for_thesis_model_output(granule_name, image,gradient_method, ML_method):
    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(14, 5), layout='constrained')
    fig, row = plt.subplots(nrows=2, ncols=3, figsize=(10, 5), layout='constrained')
    fig.tight_layout()

    titles = ['Base', 'Gradient Method', "ML Method"]
    for i in range(2):
        images = (image[i], gradient_method[i], ML_method[i])
        for j in range(3):
            a1 = sns.heatmap(images[j], ax=row[i][j], rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
            # a1.plot(border_float[0], border_float[1], color='red')
            a1.axis('off')
            # a1.set_title(titles[j])
        # Col 2
        # a2 = sns.heatmap(gradient_method, ax=row1[i], rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
        # a2.axis('off')
    row[0][0].set_title("Granule",         fontsize=22)
    row[0][1].set_title("Gradient Method", fontsize=22) 
    row[0][2].set_title("ML Method",       fontsize=22)

        # a5 = sns.heatmap(ML_method, ax=ax3, rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
        # a5.axis('off')

    rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.05, 0.35), 0.7, 0.4, fill=False, color="k", lw=2, 
        zorder=1000, transform=fig.transFigure, figure=fig
    )
    fig.patches.extend([rect])

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1) 
    # fig.suptitle(granule_name)
    # fig.subplots_adjust(top=0.88)
    numb = np.random.randint(0,1000000)
    fig.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/figs_for_thesis/debug_image{numb}.png", bbox_inches='tight')
    fig.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/figs_for_thesis/debug_image{numb}.svg", bbox_inches='tight')
    

def visualize_for_thesis_IoU_thresholds(plotting_dict):
    plotting_dict = sorted(plotting_dict, key=lambda d: d['IoU'])
    print([x['ML_area_error'] for x in plotting_dict])
    plotting_dict = {
        'base_images': [x['base_images'] for x in plotting_dict],
        'Gradient_images': [x['Gradient_images'] for x in plotting_dict],
        'ML_images': [x['ML_images'] for x in plotting_dict],
        'IoU': [x['IoU'] for x in plotting_dict],
        'ML_area_error': [x['ML_area_error'] for x in plotting_dict],
    }

    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(14, 5), layout='constrained')
    fig, row = plt.subplots(nrows=3, ncols=len(plotting_dict['base_images']), figsize=(10, 5), layout='constrained')
    fig.tight_layout()

    titles = ['Base', 'Gradient Method', "ML Method"]
    # images = (image[i], gradient_method[i], ML_method[i])
    images = (plotting_dict['base_images'], plotting_dict['Gradient_images'], plotting_dict['ML_images'])
    for i in range(3):
        for j in range(len(plotting_dict['Gradient_images'])):
            a1 = sns.heatmap(images[i][j], ax=row[i][j], rasterized=True, cmap=plt.colormaps["viridis"], cbar=False) 
            a1.get_xaxis().set_ticks([])
            a1.get_yaxis().set_ticks([])
    
    # for i in range(len(plotting_dict['Gradient_images'])):
    #     row[0][i].set_title(f"{np.round(plotting_dict['ML_area_error'][i], 2)}", fontsize=18)
        # row[0][i].set_title(f"IoU {np.round(plotting_dict['IoU'][i], 2)}", fontsize=18)
    

    row[0][0].set_ylabel("Granule",         fontsize = 14, color='black')
    row[1][0].set_ylabel('Gradient Method', fontsize = 14, color='black')
    row[2][0].set_ylabel('ML Method',       fontsize = 14, color='black')
    
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1) 
    numb = np.random.randint(0,1000000)
    # fig.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/figs_for_thesis_IoU/debug_image{numb}.png", bbox_inches='tight')
    # fig.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/figs_for_thesis_IoU/debug_image{numb}.svg", bbox_inches='tight')

    fig.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/exclusive/ML_valid/debug_image{numb}.png", bbox_inches='tight')
    fig.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/exclusive/ML_valid/debug_image{numb}.svg", bbox_inches='tight')

    # fig.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/exclusive/Gradient_valid/debug_image{numb}.png", bbox_inches='tight')
    # fig.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/exclusive/Gradient_valid/debug_image{numb}.svg", bbox_inches='tight')

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
    # csv = read_csv("test_above9.csv")
    # csv = csv.iloc[[3,5]]
    # print(csv)
    # csv = pd.read_csv("full_csv_result.csv", index_col=0)
    # csv = csv.dropna()

    # df = csv.sort_values('IoU')
    # bins =  np.arange(0, 1.0, 0.155)
    # ind = np.digitize(df['IoU'], bins)
        
    # df.groupby(ind).sample(1, replace=False)#head(2)
    # csv = df

    #----------------------------------------------------------
    # ----------- For all shared granules -------------------
    csv = pd.read_csv("csvs/full_csv_result.csv", index_col=0)
    # csv = csv[csv['IoU'] < 0.6]
    csv = csv[csv['gradient_exclusive'] == True]
    csv = csv.sample(1000, replace=False)
    # csv = csv.dropna() # Removes all exlusive granules 
    # # -------------------------------------------------------

    # # csv = csv[csv['IoU'] > 0.21]
    # # df = csv.sort_values('IoU')
    # # bins =  np.arange(0.2, 1.0, 0.15)
    # # ind = np.digitize(df['IoU'], bins)
        
    # csv = csv[(csv['ML_area_error'] > 0.85) & (csv['ML_area_error'] < 1.5)]
    # df = csv.sort_values('ML_area_error')
    # bins =  np.arange(0.85, 1.5, 0.15)
    # ind = np.digitize(df['ML_area_error'], bins)

    # csv = df.groupby(ind).sample(1, replace=False)#head(2)
    # # csv = df.groupby(ind).tail(1)
    #-----------------------------------------------------------


    # ----------- For exclusive granules ----------
    # csv = pd.read_csv("csvs/ml_exclusive_df.csv", index_col=0)
    # csv = csv[csv['area_ml'] == -1]

    # # csv = pd.read_csv("csvs/gradient_exclusive_df.csv", index_col=0)
    # csv = csv.sample(6, replace=False)
    # ----------------------------------------------

    # plotting_dict = {
    #     'base_images':[],
    #     'ML_images':[],
    #     'Gradient_images':[],
    #     'titles':[]
    # }
    plotting_list = []

    sort_by = 'ML_area_error' # 'IoU'
    print("CSV shape", csv.shape)

    for i, file in enumerate(csv.groupby('file')):
        filename = file[0]
        data = file[1]
        data.sort_values(sort_by, inplace=True) # Ensure plot has increasing IoU scores

        print(f"\n Loading {filename} \n")
        print(data['frames'].to_list())

        try:
            # im_directory = "D:\Granule_experiment_data\ALL_IMS\ALL_IMS_TOGETHER"
            im_directory = f"/Home/siv32/eto033/ims_files/{filename[:10]}/"
            ims_folders = [d for d in os.listdir(im_directory) if d[:4] == "data"]
            # print(ims_folders)
            # print("Image dir",im_directory, "\n")

            # for root, dirs, files in os.walk(im_directory): # Used if .ims files are broken down into several data1, data2, data3 subfolders for quicker analysis runtimes.
            #     if f"{filename}.ims" in files:
            #         found_file = os.path.join(root, f"{filename}.ims")
            #         print(found_file)
            #         break
            # print(f"---------> {filename[:-7]}.ims")

            if f"{filename[:-7]}.ims" in os.listdir(im_directory):
                found_file = os.path.join(im_directory, f"{filename[:-7]}.ims")
                print(found_file)


            # image_gen = fg.bioformatsGen_spesific_frames(Path(found_file), data['frames'].to_list())
            # next(image_gen)
            image_gen = fg.bioformatsGen_spesific_frames(Path(found_file), data['frames'].to_list()) # Ensure .ims file actually exists, bioformats will not throw error untill it is used by next()...

            # image_analysed_results_df = pd.read_hdf(Path(f"/Home/siv32/eto033/granule_explorer_core/experiments/ML_{filename[:10]}__1/fourier/{filename}--DEBUG.h5"), mode="r", key="fourier")
            image_analysed_results_df = pd.read_hdf(Path(f"/Home/siv32/eto033/fourier_files/gradient/{filename[:10]}/fourier/{filename}.h5"), mode="r", key="fourier")
            
        except Exception as e:
            print("\n-----")
            print(filename[:10])
            print("\n-----")
            print("\nERROR\n")
            print(e)
            break


        # print("\n\n")
        # print(data.columns)
        # print(data[:3])
        # print("\n\n")
        # data_sorted = data[1].sort_values('frames')
        for _, image_data in enumerate(image_gen):
            for row in data[data['frames'] == image_data.frame_num].sort_values(sort_by).iterrows(): # For all granules in a frame, grab image data
                # print(row[1])
                wanted_frame = row[1]['frames']
                wanted_granule_id = row[1]['granule_ids']
                if i == wanted_frame:
                    print(f"Frame {wanted_frame}")
                    
                # ----------- Making figures ----------- 
                granule_fourier = image_analysed_results_df[(image_analysed_results_df['granule_id'] == wanted_granule_id) & (image_analysed_results_df['frame'] == wanted_frame)]
                # assert granule_fourier['valid'].iloc[0] == True, f"Wanted granule with ID {wanted_granule_id} Frame {wanted_frame} is not valid!"

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

                # try:
                # ------------ Pixel ground truth ------------ 
                original_image = Image.fromarray(granule_cutout_image)
                cutout_height, cutout_width = abs(bbox_left-bbox_right), abs(bbox_bottom - bbox_top)
                _, xs_upscaled, ys_upscaled = scale_padding(original_image, (cutout_height, cutout_width), granule_fourier, NEW_MAX_HEIGHT = 1024, NEW_MAX_WIDTH = 1024)
                # assert (np.max(xs_upscaled) <= 1024) and (np.max(ys_upscaled) <= 1024), f"Number out of bounds \n x_ints: {np.max(xs_upscaled)} y_ints: {np.max(ys_upscaled)})"
                
                xs_pixels, ys_pixels = pixels_between_points(xs_upscaled, ys_upscaled)
                assert len(xs_pixels) == len(ys_pixels), f"They should have equal length {len(xs_pixels)} == {len(ys_pixels)}"

                border_image = np.zeros((1024,1024))
                # print("\n")
                # print("Max element", np.max(xs_pixels))
                # print("\n")
                for i in range(len(xs_pixels)):
                    try:
                        border_image[ys_pixels[i], xs_pixels[i]] = 1
                    except:
                        continue
                flood_fill = ski.morphology.flood(border_image, (512,512), connectivity=1)
                border_image[flood_fill == True] = 1
            
                # ------------ ML Output ------------ 
                result_mask = ML_run_model(granule_cutout_image, (x_pos_relative, y_pos_relative))

                #### visualize_better(
                ####     granule_name=f"{filename[:19]} F: {wanted_frame} ID: {wanted_granule_id} IoU: {np.round(row[1]['IoU'],2)}",
                ####     image=scale_image_add_padding(Image.fromarray(granule_cutout_image)),
                ####     gradient=scale_image_add_padding(Image.fromarray(upscaled_gradient_granule_image)),
                ####     gradient_method=border_image, 
                ####     ML_method=result_mask,
                ####     border_float=(xs_upscaled, ys_upscaled),
                ####     boder_pixel=(xs_pixels, ys_pixels),
                #### )   
                
                name_id = np.random.randint(0, 100000)
                # normed = cv2.normalize(scale_image_add_padding(Image.fromarray(granule_cutout_image)), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # base_image = cv2.applyColorMap(normed, cv2.COLORMAP_VIRIDIS) #cv2.cvtColor(scale_image_add_padding(Image.fromarray(granule_cutout_image)), cv2.COLORMAP_VIRIDIS)
                # cv2.imwrite(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/individual_granules/base_image/{name_id}.png", base_image)
                
                # result_mask = cv2.cvtColor(result_mask, cv2.COLORMAP_VIRIDIS)
                # cv2.imwrite(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/individual_granules/ml_image/{name_id}.png", result_mask)
                plt.figure(frameon=False)
                plt.imshow(scale_image_add_padding(Image.fromarray(granule_cutout_image)), cmap=plt.colormaps["viridis"])
                plt.axis('off')
                plt.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/individual_granules/base_image/{name_id}.png", bbox_inches='tight', pad_inches=0)
                plt.close()
                
                plt.figure(frameon=False)
                plt.imshow(result_mask, cmap=plt.colormaps["viridis"])
                plt.axis('off')
                plt.savefig(f"/Home/siv32/eto033/MasterProject/GE_result_analysis/figs/individual_granules/ml_image/{name_id}.png", bbox_inches='tight', pad_inches=0)
                plt.close()

                # This one for visualize_for_thesis_IoU_thresholds
                # plotting_dict = {}
                # plotting_dict['base_images'] = scale_image_add_padding(Image.fromarray(granule_cutout_image))
                # plotting_dict['Gradient_images'] = border_image
                # plotting_dict['ML_images'] = result_mask
                # plotting_dict['IoU'] = row[1]['IoU']
                # plotting_dict['ML_area_error'] = row[1]['ML_area_error']
                # plotting_list.append(plotting_dict)

    # visualize_for_thesis_IoU_thresholds(
    #     plotting_list
    # )



    # visualize_for_thesis_model_output(
    #     granule_name=f"{filename[:19]} F: {wanted_frame} ID: {wanted_granule_id} IoU: {np.round(row[1]['IoU'],2)}",
    #     image=base_images,
    #     # gradient=None,
    #     gradient_method=Gradient_images, 
    #     ML_method=ML_images,
    #     # border_float=(xs_upscaled, ys_upscaled),
    #     # boder_pixel=(xs_pixels, ys_pixels),
    # ) 

 


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
        print("----------- Done! -----------")

    main()







