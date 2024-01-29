import numpy as np
import matplotlib.pyplot as pt


# --- Make a fake granule with mean radius 10.0 and two fluctuation modes ---

angles = np.linspace(0,2*np.pi,400) # sample 400 angles, as in boundary_extraction.py
radii = 10.0 + 3.0 * np.cos(angles * 4) + 5.0 * np.cos(angles * 8) # fake granule radii


# --- Analyse the fake granule, just like granule explorer does, to get the fourier components ---

magnitudes = np.fft.rfft(radii) # Do fourier transform
print(magnitudes)

# In boundary_extraction.py, line 119, we drop the first two fourier terms, and terms higher than the 60th.
#  This is because the first Fourier term sets the mean radius (which we don't need for fluctation analysis),
#  and the second Fourier term just sets where the granule is (which we also don't care about),
#  and modes higher than 60 are so high frequency that they only represent noise in the image and so are useless.
magnitudes = magnitudes[2:60]

# we also save the mean radius and position in fourier.h5
mean_radius = np.mean(radii)

print("mean radius", mean_radius) # should be 10.0


# -------- Do the inverse fourier transform ---------
#  This should recover the radii we started out with. This is the part you need.

# We dropped the fist two modes, but we need the indicies to line up with the
# mode numbers, so here we add two zeroes to the start of the magnitudes array,
# to shift everything else over.
magnitudes = np.append( np.array([0.0+0.0j, 0.0+0.0j]), magnitudes  )

# We have add the mean_radius back in, becasue we dropped the first fourier mode 
# We want 400 samples because that was the original length of radii 
radii_1 = mean_radius + np.fft.irfft(magnitudes,400)

# ------- Do the inverse transform by hand -------

modes = np.arange(2,60) #the modes we have kept
radii_2 = np.zeros_like(radii,dtype="complex128") #array of zeroes for the radii to go in

# sum the simple waves up over all modes (remember that magnitude is still padded with two leading zeros here)
for mode in modes:
    radii_2 += magnitudes[mode] * np.exp( 2.0j * np.pi * mode * np.arange(400) / 400)
radii_2 /= len(radii) // 2 # this is part of the definition of irfft
radii_2 = radii_2.real #get rid of imaginary part: not needed

radii_2 += mean_radius # add in the mean radius

# Should be close to zero as radii_2 and radii _1 are equivalent
print("compare irfft and manual code ",sum(abs(radii_2 - radii_1)))
# Should be small but not as close to zero because we dropped some modes
print("compare original and reconstructed radii ", (abs(radii_2 - radii)).mean()) 

#plots to show that origina and reconstructed versions look the same
# pt.scatter(radii*np.cos(angles),radii*np.sin(angles))
pt.scatter(radii_1*np.cos(angles),radii_1*np.sin(angles))
# pt.scatter(radii_2*np.cos(angles),radii_2*np.sin(angles))

pt.show()
