# Tasks

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
            scp kjempefuru:/export/grellscheidfs/microscopy/2019-12-09/* .
        
        # Maybe better command. TEST THIS WITH SOME PRIVATE FILES!!!
            rsync -aHAXxv --numeric-ids --progress -e "ssh -T -c aes128-ctr -o Compression=no -x" . kjempefuru:some_local_project

            rsync -aHAXxv --numeric-ids --progress -e "ssh -T -c aes128-ctr -o Compression=no -x" . kjempefuru:/export/grellscheidfs/microscopy/2019-12-09/

            -> https://gist.github.com/KartikTalwar/4393116


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

# Pekka Questions

Chapter 3: How much background should i provide on the selected model architectures and encoders?

Chapter 4: Justification of hyperparamter value choices.

# --------------------- Kjempefuru ---------------------------------

Kjempeduru: 'top' command to see cpu usage. Google it.


# Central Conda env for central GE
### Clone this env and install the packges that 
conda activate granule_explorer_core

# Central GE version. 
cd /export/gressheid/granule_explorer/src/granule_explorer 

# Snakefile
/export/../graule_Explorer/experiments/snakefile -> Overwrite this to snakefileIMS

# Command
snakemake -d "experiment_directory"

nohop snakemake --use-conda -j 48 -d "experiment_dir" &

# -k -> Complete, continous if error arises
snakemake -k --use-conda -j 44 -d "ML_2019-10-31"

# Add source ims directory to config file


NB:
Missions, snakefile

Tannlege: 8. Juli kl 14