import h5py
import pandas as pd
import os
from shapely.geometry import Polygon
from helper_functions import get_coords
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from IPython.display import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoLocator, FuncFormatter
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

# ax.hist(fourier_df["IoU"], bins=25, linewidth=0.5, edgecolor="white")

# fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), layout='constrained')
fig = plt.figure(layout="constrained")
subfigs = fig.subfigures(nrows=2, ncols=1, )#wspace=0.07, width_ratios=[1.5, 1.])

axs0 = subfigs[0].subplots(1,1)
subfigs[0].set_facecolor('lightblue')
subfigs[0].suptitle('subfigs[0]\nLeft side')
subfigs[0].supxlabel('xlabel for subfigs[0]')


# Upper
ax1.hist(fourier_df['IoU'], 50, label=f"all experiments") 
ax1.legend(loc='lower right', fontsize=18)

# fig, axd = ax2.subplot_mosaic([['upper', 'upper', 'upper'],
#                                ['lower left', 'lower middle', 'lower right']],
#                               figsize=(12,8), layout="constrained", sharex=True, sharey=True)

axs1 = subfigs[1].subplots(1, 3)
# subfigs[1].suptitle('subfigs[1]')
# subfigs[1].supylabel('ylabel for subfigs[1')

# Lower left
ex1_df = fourier_df[fourier_df['file_name'] == '2019-10-31']
axd['lower left'].hist(ex1_df['IoU'], 50,label=f"2019-10-31", color='purple') 
axd['lower left'].legend(loc='lower right', fontsize=18)

# Lower middle
ex2_df = fourier_df[fourier_df['file_name'] == '2020-02-05']
axd['lower middle'].hist(ex2_df['IoU'], 50,label=f"2020-02-05", color='green') 
axd['lower middle'].legend(loc='lower right', fontsize=18)

# Lower middle
ex3_df = fourier_df[fourier_df['file_name'] == '2019-12-09']
axd['lower right'].hist(ex3_df['IoU'], 50,label=f"2019-12-09", color='orange') 
axd['lower right'].legend(loc='lower right', fontsize=18)

axd['upper'].text(.02, .925, 'A', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=axd['upper'].transAxes)
axd['lower left'].text(.04, .925, 'B', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=axd['lower left'].transAxes)
axd['lower middle'].text(.04, .925, 'C', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=axd['lower middle'].transAxes)
axd['lower right'].text(.04, .925, 'D', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=axd['lower right'].transAxes)

# # Set common labels
fig.supxlabel('IoU')
fig.supylabel('Count')
fig.savefig("D:/Master/MasterProject/Overleaf_figures/Chapter5/5.2/Granule_IoU_dist.svg")   