from typing import Tuple, Literal, Optional
from pathlib import Path

import astra
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tyro

from utils import load_xtekct, load_angles, load_images, PrintTableMetrics

def main(
        input_folder: Path,
        output_folder: Path,
        Lscale: int = 20,
        image_downscale: float = 1.0,
        target_resolution: Tuple[int,int,int] = (500,500,500),
        steps_per_epoch: int = 10,
        max_epochs: int = 10,
        algorithm: Optional[Literal['SIRT3D_CUDA','CGLS3D_CUDA']] = 'SIRT3D_CUDA',
        gpu_index: int = 0,
):
    output_folder.mkdir(parents=True, exist_ok=True)
    proj_data, image_indices = load_images(input_folder, image_downscale)
    print(proj_data.shape)
    
    data = load_xtekct(input_folder)
    xtekct = data['XTekCT']
    angle_map = load_angles(input_folder)
    
    vol_geom = astra.create_vol_geom(*target_resolution)

    angle_arr = np.array([angle_map[i] for i in image_indices]) * np.pi/180

    proj_geom = {
        'type':'cone',
        'DetectorSpacingX':xtekct['DetectorPixelSizeX']*image_downscale*Lscale, # L
        'DetectorSpacingY':xtekct['DetectorPixelSizeY']*image_downscale*Lscale, # L
        'DetectorRowCount':int(xtekct['DetectorPixelsX']/image_downscale), # -
        'DetectorColCount':int(xtekct['DetectorPixelsY']/image_downscale), # -
        'ProjectionAngles':angle_arr, # - 
        'DistanceOriginSource':xtekct['SrcToObject']*Lscale, # L
        'DistanceOriginDetector':(xtekct['SrcToDetector']-xtekct['SrcToObject'])*Lscale # L
    }

    # Create empty volume
    cube = np.zeros((vol_geom['GridSliceCount'], vol_geom['GridRowCount'], vol_geom['GridColCount'])) # todo check row col order
  
    # Create projection data from this
    proj_id = astra.create_sino3d_gpu(cube, proj_geom, vol_geom, returnData=False)
    astra.data3d.store(proj_id, proj_data)

    # Display a single projection image
    plt.figure(1)
    plt.imshow(proj_data[:,0,:], cmap='gray')
    plt.savefig(output_folder/'projection.png')
    plt.close()

    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict(algorithm)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['option'] = {'GPUindex': gpu_index}

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run the algorithm
    print(f'Running {max_epochs*steps_per_epoch} iterations')
    t = PrintTableMetrics(['Iteration', 'Error'], max_iter=max_epochs*steps_per_epoch)
    residual_error = np.zeros(max_epochs)
    for i in range(max_epochs):
        # Run a single iteration
        astra.algorithm.run(alg_id, steps_per_epoch)
        residual_error[i] = astra.algorithm.get_res_norm(alg_id)
        t.update({'Iteration': i*steps_per_epoch, 'Error': residual_error[i]})

    # Get the result and save
    rec = astra.data3d.get(rec_id)
    np.save(output_folder/f'reconstruction_{residual_error[-1]:.1f}.npy', rec)

    plt.figure(2)
    plt.imshow(rec[:,:,vol_geom['GridColCount']//2])
    plt.savefig(output_folder/'slice.png')
    plt.close()


    plt.figure(4)
    plt.plot(residual_error)
    plt.savefig(output_folder/'error.png')
    plt.close()


    out_dir = output_folder/Path('./slices')
    out_dir.mkdir(exist_ok=True, parents=True)
    for i in range(0,rec.shape[2],rec.shape[2]//64):
        cv.imwrite(str(out_dir/f'slice_{i:03d}.png'), ((rec[:,:,i]-rec.min()) / (rec.max()-rec.min())*255).astype(np.uint8))

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)


if __name__ == '__main__':
    tyro.cli(main)