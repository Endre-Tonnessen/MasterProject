# Tasks

# Notes: 12.03.2024
# TODO: Add filters from Granule Explorer into GE Viz comparision. Filters passrate, tensions, etc. Allows us to compare results on the relevant granules. Removes granules of one pixels...

# Do IoU to find how similar they are, and check for glitching (Large deviantion of IoU)

# Compare granule by granule, id should be the same (check center to verify that the granule is avtually the granule)
Check if both suface tentions error bars are within both ranges. 


# Modify model architectures - Maybe
    Training data consists of 1 color-channel images (Grayscale, (h,w)). The segementation models want 3 channel images, RGB. Currently there is a conversion between the formats, however this is technically redundant. 

    Proposal: Modify input layer of YOLOv8 to handle (H,W,1) size images. Modify training functions to account for this.  

# Possible problem with np.argmax in sample_at_angle() - FIXED - Problem was interpolation, order was 1 instead of 0.
In the event of more than one possible boundry, it seems to return the last boundry and not the first one. (We want the first boundry)
See file:///C:/Users/Endre/Desktop/boundry_images/5.html

# Upscaling Changes - DONE
    Instead of upscaling a granule to 1024x1024, irrespective of previous image shape. Upscale to closest multiple, ensuring the original aspect ratio is preserved.
    Any space left can be padded, or we simply ignore it allowing YOLOv8 to do the scaling.

## Granule Explorer Integration - DONE

    Almost done, one last scaling bug to fix.

    Discuss `angle_sweep_scaling()` line 281, see line 340, `get_peak_location()` this needs to be scaled back down to original granule size.

        In `boundry_extraction.py` line 65, add a new _BoundaryExtractionGradient method for using a machine learning model.
        This call will return data in the same format as the others. 
        Must do some post-processing to get correct fourier terms from pixel mask output.

        Maybe modify the top-level `manager.py` to include arguments for `image_processing` method. See line 144 and 148 in `process_image.py` for implementation.
        NOTE: ADDING OFFSET IS CORRECT!s
        TODO: order parameter in angle_sweep function on function map_cooridnates(order=0) to prevent cubic interlation. Cubic will interpolate, avarage between pixel values.

        When integrating the model, use onnex export format? Might be compatible with all model architectures, allowing tensorflow to run any model without installing all corresponding model frameworks. Could make integration easier.

# Get more .ims files with corresponding .h5 analysis files - DONE
Need to make a larger dataset for training. Current training set is composed from only one .ims file. FIND .TXT FILE WITH EXPERIMENT NAME DATA Jack mentioned.
    -> Go over how to ssh into UIB servers and download files.
    -> Unable to find file containing experiment names and their corresponding .ims files.

        ssh eto033@login.uib.no # Microsoft
        ssh endret@login.ii.uib.no # II password
        ssh endret@kjempefuru.cbu.uib.no # II password
        scp kjempefuru:/export/grellscheidfs/microscopy/2020-02-05/2020-02-05_15.11.08--NAs--T1354-GFP_Burst.ims ./data 

        # From local console, run to grab ALL files in external folder and put it at current cmd location.
        scp kjempefuru:/export/grellscheidfs/microscopy/2020-02-05/* .

        # New files | USE THIS ONE
            scp kjempefuru:/export/grellscheidfs/microscopy/2019-10-31/2019-10-31_12.18.43--NControlLongB--T1015-Burst.h5 .
            scp kjempefuru:/export/grellscheidfs/microscopy/2019-10-31/ .

# What should the model predict? - DONE
    -> Currently the model will make a prediction no matter how the granule looks. This can lead to undesirable results, we may not always want to classify every granules? Any we should exclude?
*No changes needed*

# Plan v2

* Remake dataset with new border scaling offset added.
* Retrain models
* Integrate model into Granule Explorer
    * Should be 'easy'. Make a new boundry extraction method that uses a ML model. 
    * Outputs image with just the border pixels having values of 1. Rest of image is 0.
    * Might have to redo parts of the get_fourier_terms() function to account for up- and downscaling of image and border.
        * Also a question if current boundry storing method supports funny-shaped granules at all. Jack will look into this.


## Upscaling offset bug - DONE
Create a small 10x10 picture. Create a square of 400 points. Upscale and calculate where the points should land and which pixel that is.
Verify that they are where they should be in the picture.

We solved this. Granule explorer uses map_coordinates(order=0) from scipy that treats pixels as having their interger coordinates in the center. Pushing the entire image by -1/2, introducing weird offset settings.

# get_intersected_pixels() - DONE
Edge case when line is directly in the middle of pixels. Which ones do we pick? 
    - Suggestion: Add a small value 0.0001 prevent this. Will introduce some bias? Need bias anyway?


# Thesis

Paper: https://arxiv.org/abs/2305.09972
-> Have a look at feature maps, displaying how the model views the image at different layers.
    -> Can explain class confusion. Ofcourse i only have 1 class currently...



# Kjempefuru

python ../../granule_explorer_core/workflow/manager.py process-image /export/grellscheidfs/microscopy/2020-02-05/2020-02-05_15.11.08--NAs--T1354-GFP_Burst.ims .


# Plots 
Remove background color, should be white. Rmoemove lines.

Increase label text to be size of thesis text.

Remove titles, Add A and B, 

Figure 4.5, see paper. Remove perspctive thing, remove title, increase fonts size, Add A, B, C ticks.

Font should be black.

Fig 4.7: Make into heatmap, which makes difference between 0 and low amounts of granules more clear. 
Currently it seems that there are more granules in the edge cases then there actually are. 


# Jack
Finish full draft of chapter 4.

# Pekka
Vacation: when are you on vacation?
    -> 28. Juni og hele Juli

Er Juli vurderingsfri?
Jack has an external supervisor from Germany who is available in July. 

Section 4.2 and section 5. Is the current model evaluation and selection reasonable?
    -> Yes!


# Stanislav
Send mail asking about the amount of storage space a student has on birget.
Ask about transfering large amounts of data between kjempefuru and birget. (Need all the .ims files and their corresponding gradient analysis results.)

/export/grellscheidfs/microscopy

## Chapter 5 ML results
For all training iou plots, add the legend inside the plot. There is usually room for them. Val iou can be as normal.
https://plotly.com/python/legend/


# NREC: https://dashboard.nrec.no/dashboard/project/
Username: 'endre.tonnessen@student.uib.no'
Password: 'zwLsfsgGl97QhedC'


# Copying files

Sodium Arsinite 2020- the one i have used

2019-10-31 - Arsinite

2019-12-09 - Ctrlomozone


[Implement new GE version.]