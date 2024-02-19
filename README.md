# Plan

# Upscaling Changes - TODO
Instead of upscaling a granule to 1024x1024, irrespective of previous image shape. Upscale to closest multiple, ensuring the original aspect ratio is preserved.
Any space left can be padded, or we simply ignore it allowing YOLOv8 to do the scaling.

## Granule Explorer Integration - TODO

In `boundry_extraction.py` line 65, add a new _BoundaryExtractionGradient method for using a machine learning model.
This call will return data in the same format as the others. 
Must do some post-processing to get correct fourier terms from pixel mask output.

Maybe modify the top-level `manager.py` to include arguments for `image_processing` method. See line 144 and 148 in `process_image.py` for implementation.
NOTE: ADDING OFFSET IS CORRECT!s
TODO: order parameter in angle_sweep function on function map_cooridnates(order=0) to prevent cubic interlation. Cubic will interpolate, avarage between pixel values.

When integrating the model, use onnex export format? Might be compatible with all model architectures, allowing tensorflow to run any model without installing all corresponding model frameworks. Could make integration easier.

# Get more .ims files with corresponding .h5 analysis files - TODO
Need to make a better dataset for training. Current training set is composed from only one .ims file. FIND .TXT FILE WITH EXPERIMENT NAME DATA Jack mentioned.

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
