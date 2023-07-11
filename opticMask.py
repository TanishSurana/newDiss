# we have mt and mt+1 masks
# we have it and it+1 images (frames)

# use it and it+1 images to generate optical flow ot
# ot will tell us how the pexels are moving in form of vectors between it and it+1. 
# now we can use this ot and select only the pixels in mask mt, to predict the next mask: m*t+1

# this m*t+1 in theory should be equivalent to mt+1.

# we will compare mt+1 and m*t+1, to see how good this or any optical flow information is in predicting mask aka mirror movement. 

# to compare basic will be IoU
