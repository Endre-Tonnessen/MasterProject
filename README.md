# Plan




## Granule Explorer Integration

In `boundry_extraction.py` line 65, add a new _BoundaryExtractionGradient method for using a machine learning model.
This call will return data in the same format as the others. 
Must do some post-processing to get correct fourier terms from pixel mask output.

Maybe modify the top-level `manager.py` to include arguments for `image_processing` method. See line 144 and 148 in `process_image.py` for implementation.