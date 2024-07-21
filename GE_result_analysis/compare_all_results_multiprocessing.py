import h5py
import pandas as pd
import os
from shapely.geometry import Polygon
from helper_functions.helper_functions import get_coords
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from IPython.display import Image
from multiprocessing import Process, Queue

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

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# def analyze_h5_files(ML_PATH: str, GRADIENT_PATH: str, verbose=False):
def analyze_h5_files(filenames, verbose=False):
    ML_PATH: str = filenames[0]
    GRADIENT_PATH: str = filenames[1]
    # --------------- Load data  ---------------
    fourier_pd_gradient = pd.read_hdf(
        GRADIENT_PATH, key="fourier", mode="r"   
    )
    fourier_pd_ml = pd.read_hdf(
        ML_PATH, key="fourier", mode="r"
    )
    fourier_pd_gradient_valid = fourier_pd_gradient[(fourier_pd_gradient['valid'] == True)]
    fourier_pd_ml_valid = fourier_pd_ml[(fourier_pd_ml['valid'] == True)]
    # assert fourier_pd_gradient_valid['frame'].max() == fourier_pd_ml_valid['frame'].max(), f"Unequal frames! Was {fourier_pd_gradient_valid['frame'].max()} != {fourier_pd_ml_valid['frame'].max()}"
    min_frames = np.min((fourier_pd_gradient_valid['frame'].max(), fourier_pd_ml_valid['frame'].max()))
    # assert (fourier_pd_gradient_valid['frame'].max() == fourier_pd_ml_valid['frame'].max()), f"{fourier_pd_gradient_valid['frame'].max()} != {fourier_pd_ml_valid['frame'].max()}"
    # --------------- Valid granules ---------------
    print("\n")
    print(f"Current: {pathlib.Path(GRADIENT_PATH).stem}")
    nr_valid = fourier_pd_gradient_valid[fourier_pd_gradient_valid['frame'] <= min_frames]['granule_id'].unique() 
    print(f"Valid granules (gradient) | {nr_valid.size}")
    nr_valid_ml = fourier_pd_ml_valid[fourier_pd_ml_valid['frame'] <= min_frames]['granule_id'].unique()
    print(f"Valid granules (ML) | {nr_valid_ml.size} \n") 

    granules_in_common_total = np.intersect1d(nr_valid, nr_valid_ml).size
    gradient_exclusive_total = np.setdiff1d(nr_valid, nr_valid_ml).size
    ml_exlusive_total = np.setdiff1d(nr_valid_ml, nr_valid).size

    print("Union", granules_in_common_total)
    print("Gradient Exclusive", gradient_exclusive_total)
    print("ML Exclusive", ml_exlusive_total)
    
    # --------------- Do comparison ---------------
    IoU = []
    area_gradient_list = []
    area_ml_list = []
    granule_ids = [] 
    frames = []
    gradient_mean_intensity, gradient_mean_radius, gradient_major_axis, gradient_minor_axis, gradient_eccentricity = [],[],[],[],[]
    ml_mean_intensity, ml_mean_radius, ml_major_axis, ml_minor_axis, ml_eccentricity = [],[],[],[],[]
    common, ml_ex, gradient_ex = [],[],[]

    for frame_id in range(min_frames +1): # For all frames
        granules_in_gradient_frame = fourier_pd_gradient_valid[fourier_pd_gradient_valid['frame'] == frame_id]
        valid_granule_gradient_ids = granules_in_gradient_frame['granule_id'].unique() # All granules

        granules_in_frame_ml = fourier_pd_ml_valid[fourier_pd_ml_valid['frame'] == frame_id]
        # valid_granule_ids_ml = granules_in_frame_ml['granule_id'].unique() # All granules
        # print(valid_granule_ids)
        # print(valid_granule_ids_ml)
        # assert np.array_equal(valid_granule_ids, valid_granule_ids_ml), "Should always be equal"
        
        # TODO: Run this to ensure granule ids are the same for both files!!
        # assert granule_f_gradient_terms['granule_id'].unique() == granule_f_terms_ml['granule_id'].unique(), f"Granule ID's are not the same! Was \n {granule_f_gradient_terms['granule_id'].unique()} \n {granule_f_terms_ml['granule_id'].unique()}"


        granules_in_common = np.intersect1d(granules_in_gradient_frame['granule_id'].unique(), granules_in_frame_ml['granule_id'].unique())
        gradient_exclusive = np.setdiff1d(granules_in_gradient_frame['granule_id'].unique(), granules_in_frame_ml['granule_id'].unique())
        ml_exlusive = np.setdiff1d(granules_in_frame_ml['granule_id'].unique(), granules_in_gradient_frame['granule_id'].unique())
        # print(f"Common {granules_in_common}")
        # print(f"Gradient Exclusive {gradient_exclusive}")
        # print(f"ML exclusive {ml_exlusive}")

        # for granule_id in valid_granule_gradient_ids: # For all granules in frame       # TODO: Currently only considering granules that are in Gradient! This leaves unique granules that are in ML out!!
        for granule_id in granules_in_common: # For all common granules in frame
            
            granule_f_gradient_terms = granules_in_gradient_frame[granules_in_gradient_frame['granule_id'] == granule_id]
            granule_f_terms_ml = granules_in_frame_ml[granules_in_frame_ml['granule_id'] == granule_id]

            if granule_f_terms_ml.size == 0 :#or (granule_f_gradient_terms.iloc[0][['x','y']].tolist() != granule_f_terms_ml.iloc[0][['x','y']].tolist()):
                continue
            elif granule_f_terms_ml['mean_radius'].iloc[0] == 0 or granule_f_gradient_terms['mean_radius'].iloc[0] == 0:
                if verbose:
                    print("\nMean radius was 0!\n")
                    # return # TODO: Test if this is ever triggered!!
                continue
            # Compare x,y positions to verify comparisons happen with correct granules, irrespective of their id's
            if (granule_f_gradient_terms.iloc[0][['x','y']].tolist() != granule_f_terms_ml.iloc[0][['x','y']].tolist()):
                if verbose:
                    print(f"\n Granules not in same position! \n ML F:{granule_f_terms_ml['frame'].iloc[0]} Id: {granule_f_terms_ml['granule_id'].iloc[0]} Gradient F:{granule_f_gradient_terms['frame'].iloc[0]} Id: {granule_f_gradient_terms['granule_id'].iloc[0]}  \n{np.round(granule_f_gradient_terms.iloc[0][['x','y']].tolist(), 4)} != {np.round(granule_f_terms_ml.iloc[0][['x','y']].tolist(), 4)}")
                # return # TODO: Test if this is ever triggered!!
                continue

            xs, ys = get_coords(granule_f_gradient_terms, get_relative=True)
            xy = np.vstack((xs,ys)).T
            area_gradient = Polygon(xy)

            xs_ml, ys_ml = get_coords(granule_f_terms_ml, get_relative=True) 
            xy = np.vstack((xs_ml,ys_ml)).T 
            area_ml = Polygon(xy) 

            if not area_gradient.is_valid or not area_ml.is_valid:
                if verbose:
                    print(f"Invalid Gradient: {area_gradient.is_valid} ML: {area_ml.is_valid}") # TODO: Need to handle invalid geometries. Maybe add to csv, but note as 'invalid_geometry'
                continue 
            intersection = area_gradient.intersection(area_ml).area / area_gradient.union(area_ml).area

            IoU.append(intersection)
            area_gradient_list.append(area_gradient.area)
            area_ml_list.append(area_ml.area)
            granule_ids.append(granule_f_gradient_terms.iloc[0]['granule_id'])
            frames.append(granule_f_gradient_terms.iloc[0]['frame'])
            gradient_mean_intensity.append(granule_f_gradient_terms.iloc[0]['mean_intensity'])
            gradient_mean_radius.append(granule_f_gradient_terms.iloc[0]['mean_radius'])
            gradient_major_axis.append(granule_f_gradient_terms.iloc[0]['major_axis'])
            gradient_minor_axis.append(granule_f_gradient_terms.iloc[0]['minor_axis'])
            gradient_eccentricity.append(granule_f_gradient_terms.iloc[0]['eccentricity'])      

            ml_mean_intensity.append(granule_f_terms_ml.iloc[0]['mean_intensity'])
            ml_mean_radius.append(granule_f_terms_ml.iloc[0]['mean_radius'])
            ml_major_axis.append(granule_f_terms_ml.iloc[0]['major_axis'])
            ml_minor_axis.append(granule_f_terms_ml.iloc[0]['minor_axis'])
            ml_eccentricity.append(granule_f_terms_ml.iloc[0]['eccentricity'])  

            common.append(True) 
            ml_ex.append(False)  
            gradient_ex.append(False) 
        
        for granule_id in gradient_exclusive:
            granule_f_gradient_terms = granules_in_gradient_frame[granules_in_gradient_frame['granule_id'] == granule_id]

            if granule_f_gradient_terms.size == 0 :#or (granule_f_gradient_terms.iloc[0][['x','y']].tolist() != granule_f_terms_ml.iloc[0][['x','y']].tolist()):
                raise Exception(f"Granule does not exist? Frame: {frame_id} ID: {granule_id}")
            
            xs, ys = get_coords(granule_f_gradient_terms, get_relative=True)
            xy = np.vstack((xs,ys)).T
            area_gradient = Polygon(xy)
            assert granule_f_gradient_terms.iloc[0]['granule_id'] == granule_id
            assert granule_f_gradient_terms.iloc[0]['frame'] == frame_id
            
            IoU.append(None)
            area_ml_list.append(None)
            area_gradient_list.append(area_gradient.area)
            granule_ids.append(granule_id)
            frames.append(frame_id)
            gradient_mean_intensity.append(granule_f_gradient_terms.iloc[0]['mean_intensity'])
            gradient_mean_radius.append(granule_f_gradient_terms.iloc[0]['mean_radius'])
            gradient_major_axis.append(granule_f_gradient_terms.iloc[0]['major_axis'])
            gradient_minor_axis.append(granule_f_gradient_terms.iloc[0]['minor_axis'])
            gradient_eccentricity.append(granule_f_gradient_terms.iloc[0]['eccentricity']) 

            ml_mean_intensity.append(None)
            ml_mean_radius.append(None)
            ml_major_axis.append(None)
            ml_minor_axis.append(None)
            ml_eccentricity.append(None)  

            common.append(False) 
            ml_ex.append(False)  
            gradient_ex.append(True) 
        
        for granule_id in ml_exlusive:
            granule_f_terms_ml = granules_in_frame_ml[granules_in_frame_ml['granule_id'] == granule_id]

            if granule_f_terms_ml.size == 0 :#or (granule_f_gradient_terms.iloc[0][['x','y']].tolist() != granule_f_terms_ml.iloc[0][['x','y']].tolist()):
                raise Exception(f"Granule does not exist? Frame: {frame_id} ID: {granule_id}")
                
            xs_ml, ys_ml = get_coords(granule_f_terms_ml, get_relative=True) 
            xy = np.vstack((xs_ml,ys_ml)).T 
            try:
                area_ml = Polygon(xy).area
            except Exception as e:
                print("Ivalid geometry")
                print(e)
                area_ml = -1
            
            IoU.append(None)
            area_gradient_list.append(None)
            area_ml_list.append(area_ml)
            granule_ids.append(granule_id)
            frames.append(frame_id)
            gradient_mean_intensity.append(None)
            gradient_mean_radius.append(None)
            gradient_major_axis.append(None)
            gradient_minor_axis.append(None)
            gradient_eccentricity.append(None)       

            ml_mean_intensity.append(granule_f_terms_ml.iloc[0]['mean_intensity'])
            ml_mean_radius.append(granule_f_terms_ml.iloc[0]['mean_radius'])
            ml_major_axis.append(granule_f_terms_ml.iloc[0]['major_axis'])
            ml_minor_axis.append(granule_f_terms_ml.iloc[0]['minor_axis'])
            ml_eccentricity.append(granule_f_terms_ml.iloc[0]['eccentricity']) 

            common.append(False) 
            ml_ex.append(True)  
            gradient_ex.append(False) 

        
    df = pd.DataFrame({
        "granule_ids": granule_ids,
        "frames": frames,
        "area_gradient":area_gradient_list,
        "area_ml":area_ml_list,
        "IoU":IoU,
        "gradient_mean_intensity": gradient_mean_intensity,
        "gradient_mean_radius": gradient_mean_radius,
        "gradient_major_axis": gradient_major_axis,
        "gradient_minor_axis": gradient_minor_axis,
        "gradient_eccentricity": gradient_eccentricity,

        "ml_mean_intensity": ml_mean_intensity,
        "ml_mean_radius": ml_mean_radius,
        "ml_major_axis": ml_major_axis,
        "ml_minor_axis": ml_minor_axis,
        "ml_eccentricity": ml_eccentricity,
        
        "granules_is_common": common,
        "ml_exlusive": ml_ex,  
        "gradient_exclusive": gradient_ex 
    })
    df['Gradient_valid'] = nr_valid.size
    df['ML_valid'] = nr_valid_ml.size
    df['Granules_in_common_count'] = granules_in_common_total
    df['Gradient_exclusive_count'] = gradient_exclusive_total
    df['ML_exlusive_count'] = ml_exlusive_total
    df.to_csv(f"D:\Master\MasterProject\GE_result_analysis\comparison_results/{pathlib.Path(GRADIENT_PATH).stem}.cvs")

    del fourier_pd_gradient
    del fourier_pd_ml

def get_item_from_Queue(image_files_queue: Queue):
    while image_files_queue.qsize() > 0:
        filename = image_files_queue.get()
        print("Started with", filename[1])
        analyze_h5_files(filename[0], filename[1], verbose=False)

# def get_item_from_Queue(image_files_queue: Queue):
#     print("Running")
#     while image_files_queue.qsize() > 0:
#         filename = image_files_queue.get()
#         print("Started with", filename[1])
#         analyze_h5_files(filename[0], filename[1], verbose=False)


# def multi_process(files_to_analze):
#     image_files_queue = Queue() 
#     for filenames in files_to_analze: # Populate Queue with filenames
#         image_files_queue.put(filenames)

#     print(image_files_queue.qsize())

#     processes = [Process(target=get_item_from_Queue, args=(image_files_queue,)) for _ in range(6)]
#     print(processes)

#     for process in processes: 
#         process.start() # Start

#     for process in processes:
#     process.join()  # Stop


# if __name__ == "__main__":
#     def main():
#         generate_granule_cutout_images()

#     main()




















