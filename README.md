# NeedleSegmenter
NeedleSegmenter module in 3Dslicer

This 3D Slicer module locates the needle tip in MRI images using a custom built algorithim using phase unwrapping and other techniques. 

Prerequisites to run the module:
1. Slicer has to be built from source code 

2. build the following cli module inside your build tree. Code for the phase unwrapping module is provided by Junichi Tokuda at :
github.com/tokjun/PhaseUnwrapping

3. Add the NeedSegmenter module as a scripted python module

4. Select the magnitude and phase image of the desired volume and select which slice to preform the algorithim on. 

