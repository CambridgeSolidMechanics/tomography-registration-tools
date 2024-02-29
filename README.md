# tomography-registration-tools
Python tools for manipulation of volume data.

The two main purposes of the codebase are
1. Create synthetic volume data for use in volume registration
2. Apply arbitrary deformation fields to either (i) synethetic, or (ii) real tomographic data.

## Creation of synthetic data
Along with resolution, we can set the length-scale of texture:
![image](https://github.com/igrega348/tomography-registration-tools/assets/40634853/c190c8cc-0e8c-4439-8b88-9814fb5a2ec9)

## Applying deformation field
Arbitrary deformation field can be applied to both synthetic volume data or real tomographic data. 
Here is an example of applying localised deformation field to a volume file and reconstructing that field using elastix <sup>[1](#myfootnote1)</sup> non-rigid registration:
![image](https://github.com/igrega348/tomography-registration-tools/assets/40634853/75b1122d-bed9-49f3-bbf6-eca815c8f910)


## Setting up environment
It is advisable to use conda

Create environment 
``conda env create -f environment.yml``

## Examples
See the examples in folder **examples**

Launch interactive notebook:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/CambridgeSolidMechanics/tomography-registration-tools.git/HEAD?labpath=examples%2Fdemo.ipynb)

## References
<a name="myfootnote1">1</a>: https://github.com/SuperElastix/elastix
