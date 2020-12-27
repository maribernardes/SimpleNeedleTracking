# NeedleSegmenter
NeedleSegmenter module in 3Dslicer

This 3D Slicer module locates the needle tip from Magnitude and phase MR images. 

Prerequisites to run the module:

1. Slicer has to be built from source code. Follow the instructions in the following link : https://slicer.readthedocs.io/en/latest/developer_guide/build_instructions/index.html 
    > Python 3.6.X needed in the build  

2. Build the the phase unwrapping CLI module inside your build tree. Code for the phase unwrapping module is provided by Junichi Tokuda at :
github.com/tokjun/PhaseUnwrapping

3. Add the NeedSegmenter module as a scripted python module. 

4. Install required python libraries in Slicer's python using pip module. Libraries required can be found in requirements.txt

5. Select the magnitude and phase image of the desired volume and select which slice to preform the algorithim on. 

6. Use the realtime to auto update the needle tip position from the selected node viewer 
