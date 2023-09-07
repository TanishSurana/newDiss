# Dissertation files

Note: This is a bit cluttered, as I have done a lot of experiments
Note: I cannot upload the dataset I created by I am sumbitting a python file, that was used to generate my dataset. The dataset size is 35 GB so cannot
Here is a summary of what files are there and how to run each experiment and analysis I did. 

### dataset created
Its on the drive: https://drive.google.com/drive/folders/1m6K6v_Yy1v2ZH3ask_KQUSvvRVbL3_ic?usp=share_link
Also, it has the Python files used to create it and for analysis. Basically, everything related to data collection

### Pretrained Model paths
you will need to download the pretrained model path before you run it. Googling the model name or the paper name will get you the links to the .pth file

### Optical flow analysis of RAFT model
opticMask.py is used to test the RAFT model, see the runcode.txt file, which has the parameters names and values to run it


### Refinement using Segmentation mask
The train_base.py, train_optic.py and train_sam.py are used for training
But first you need to get the optical features and VMD_mask from the drive link above. Then set the path and model you want to train in the .py file






