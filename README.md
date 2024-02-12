# Plan

# Upscaling Changes
Instead of upscaling a granule to 1024x1024, irrespective of previous image shape. Upscale to closest multiple, ensuring the original aspect ratio is preserved.
Any space left can be padded, or we simply ignore it allowing YOLOv8 to do the scaling.

## Upscaling offset bug
Create a small 10x10 picture. Create a square of 400 points. Upscale and calculate where the points should land and which pixel that is.
Verify that they are where they should be in the picture.

## Granule Explorer Integration

In `boundry_extraction.py` line 65, add a new _BoundaryExtractionGradient method for using a machine learning model.
This call will return data in the same format as the others. 
Must do some post-processing to get correct fourier terms from pixel mask output.

Maybe modify the top-level `manager.py` to include arguments for `image_processing` method. See line 144 and 148 in `process_image.py` for implementation.