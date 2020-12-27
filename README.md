# NeedleSegmenter
NeedleSegmenter module in 3Dslicer

This 3D Slicer module locates the needle tip from Magnitude and phase MR images. 

INSTALL:

1. Slicer has to be built from source code. Follow the instructions in the following link : https://slicer.readthedocs.io/en/latest/developer_guide/build_instructions/index.html 
    > Python 3.6.X needed in the build  

2. Build the the phase unwrapping CLI module inside your build tree. Code for the phase unwrapping module is provided by Junichi Tokuda at :
github.com/tokjun/PhaseUnwrapping

3. Add the NeedSegmenter module as a scripted python module. 

4. Install required python libraries in Slicer's python using pip module. Libraries required can be found in python_requirements.txt

USAGE: 

1. Select the magnitude and phase image of the desired volume. The mask is automatically generated from the magnitude image. 

2. Four modes are available :
 A) Manual Segmentation using slice slider (advanced)
 B) Segment Needle from the slice shown in the scene view 
 C) Simulated Tracking automates needle tracking while scrolling through different slices
 D) Live Tracking Protocol used in conjuction with SRC protocol. Adjust the FPS accordingly. 
 
Demo of the module can be found under Preview.mp4  

Known bugs
1. The option to select the scene view of choice is not available. Only the Green scene is active however it can be changed to 
any view (Axial, Sagittal, Coronal).  

