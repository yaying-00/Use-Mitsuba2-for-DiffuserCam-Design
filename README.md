# Mitsuba2-Differentiable-Rendering

In typical computational cameras, the optical system is designed first and fixed. Then the parameters in the image reconstruction algorithm are tuned to reproduce the image well. In contrast to this sequential design method, we hope to jointly optimize the optical system (e.g., the physical shape of the diffuser) and the parameters of the reconstruction model. Therefore, we need to build a fully-differentiable simulation model, mapping the source data to the reconstructed one. The model should include **a differentiable renderer**. It can be used to generate the point spread function given an arbitrary diffuser surface, or alternatively it can be used to directly generate the simulated measurement given the input 2D or 3D dataset. 

[Mitsuba 2](http://www.mitsuba-renderer.org/) is a physically-based rendering tool, which can be used as a differentiable renderer in the end-to-end optimization.

Paper and conference video can be found [here](http://rgl.epfl.ch/publications/NimierDavidVicini2019Mitsuba2).

[Mitsuba 2 documentation](https://mitsuba2.readthedocs.io/en/latest/src/getting_started/intro.html) is a good resource for installation, tutorials and references.

This repository contains a few examples for optimizing a diffuser. The slides which contains the corresponding optimization results can be found [here](https://docs.google.com/presentation/d/1_vz62zo_9vgIiwe38vPrAgo-MeKZbOf1P657d3Er7dU/edit#slide=id.g82412729a5_0_267). 

## Installation and Compiling

First install Mitsuba 2 according to the intructions in the [documentation](https://mitsuba2.readthedocs.io/en/latest/src/getting_started/cloning.html).

Differentiation of shapes in Mitsuba 2 requires a plugin that is based on the article [Reparameterizing discontinuous integrands for differentiable rendering](http://rgl.epfl.ch/publications/Loubet2019Reparameterizing). To run the examples, you need to change to the `pathreparam-optix7` branch of Mitsuba 2 which contains `src/integrators/path-reparam.cpp`.

After Mitsuba is compiled, remember to run the `setpath.sh/bat` script to configure environment variables every time you want to run Mitsuba.

```
cd <..mitsuba repository..>
source setpath.sh
```

## Running the examples

The code can then be executed with Python. For example,

```
python copper_optm.py
```

1. `copper_optm.py`: Optimize the height profile of a copper panel.
   
2. `diffuser_optm.py`: Optimize the height profile of a glass panel.
   
   This file contains three tasks:

   * `task = 'plain2bumpy'`    
        Optimize a plain glass panel to be a bumpy diffuser.
   * `task = 'bumpy2plain'`     
        Optimize a bumpy diffuser to be a plain surface.
   * `task = 'bumpy2bumpy'`    
        Optimize a bumpy diffuser to be another bumpy diffuser.

3. `diffuser_optm_smooth.py`: Optimize the height profile of a glass panel. Add Gaussiam smoothing between optimizations.     
    This file also contains three tasks as explained above. 