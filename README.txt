*********************************************************************************
*
* This toolbox is mostly dedicated to:
* - Comparing various parallel implementations of a denoising code
*   in Matlab. We compare: 
*   Standard Matlab, GPU Matlab, C with OpenMP, and C with CUDA on
*   a powerful workstation.
* - Provide hints on how to program a CUDA file interfaced with Matlab.
* - Provide hints on how to setup a decent environment for GPU programming. 
*
* Developer: Pierre Weiss, January 2017.
*
**********************************************************************************

To use the toolbox, follow the instructions given in the associated html file. 
Once the environment is correctly setup, open matlab and type:

mex_all

Then you can try the different functions by typing:

Script_Test
