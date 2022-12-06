# NeedleTracking
NeedleTracking module in 3D Slicer

This 3D Slicer module locates the needle tip from 2D MR images
Input requirement: 
    Magnitude/Phase image or Real/Imaginary image. 
Uses scikit unwrapping algorithm


INSTALL:
1. Add the NeedTracking module as a scripted python module. 
2. Install required python libraries in 3D Slicer's python using the Python Interactor interface. 
Libraries required can be found in python_requirements.txt

USAGE: 
1. Define the input mode
Two input modes are excepted:
    Magnitude and phase
    Real and imaginary
2. Select the image pair
3. Select the scene view where the needle will be tracked
4. Press button for tracking
Two tracking options are available
    Track once with current image in the scene view ("Detect Needle" button)
    Cyclic track with timer defined by update rate ("Start/Stop Live Tracking" buttons) 

