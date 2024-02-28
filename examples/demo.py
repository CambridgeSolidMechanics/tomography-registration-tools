
import os
import sys
from pathlib import Path
import torch

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
from t_core import volume_tools as vt

def deform_volumes(make=True):
    """Apply deformation field to volume

    Args:
        make (bool, optional): If True, make new volume. 
            Otherwise import existing the volume. Defaults to True.
    """
    if make:
        vol = vt.make_volume((100,120,150), speckle_size=2, convolution_kernel=3)
    else:
        fn = Path('./vol_def.raw')
        res = list(map(int, Path('./resolution_vol.txt').read_text().split('x')))
        vol = vt.load_volume(fn, res)

    vt.plot_acf(vol)

    # choose a deformation field
    sigma_vox = 10 # voxels
    sigma_x = sigma_vox / vol.shape[0] * 2
    sigma_y = sigma_vox / vol.shape[1] * 2
    sigma_z = sigma_vox / vol.shape[2] * 2
    A_vox = 3 # voxels
    # note for this type of field, the requirement for compatibility is
    # A_vox < sigma_vox*exp(-0.5) ~ 1.65*sigma_vox
    A_z = A_vox / vol.shape[2] * 2
    z_func = lambda x,y,z: z + A_z*torch.exp(-0.5*(x**2/sigma_x**2 + y**2/sigma_y**2 + z**2/sigma_z**2))

    undef_vol = vt.deform_general(
        vol, 
        x_func=lambda x,y,z: x,
        y_func=lambda x,y,z: y,
        z_func=z_func
    )
    vt.plot_undef_def_slices(vol, undef_vol)

    name = f'vol'
    fn = Path(f'./{name}_undef.raw')
    vt.save_volume(fn, undef_vol)

    if make:
        fn = Path(f'./{name}_def.raw')
        vt.save_volume(fn, vol)
        Path(f'./resolution_{name}.txt').write_text('x'.join(map(str, vol.shape)))

if __name__=='__main__':
    """Demonstration of
    1. How to make a synthetic volume and deform it
    2. How to load an existing volume and deform it
    """
    deform_volumes(make=True)
    deform_volumes(make=False)