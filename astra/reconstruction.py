# %%
import astra
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv
from utils import load_xtekct, load_angles, load_images, PrintTableMetrics
import tyro

def main(
        input_folder: Path,
        output_folder: Path,
        Lscale: int = 20,
        image_downscale: float = 1.0,
):
    output_folder.mkdir(parents=True, exist_ok=True)
    proj_data, image_indices = load_images(input_folder, image_downscale)
    print(proj_data.shape)
    
    data = load_xtekct(input_folder)
    print(data.keys())
    xtekct = data['XTekCT']
    print(xtekct.keys())
    angle_map = load_angles(input_folder)
    
    vol_geom = astra.create_vol_geom(500, 500, 2000)

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
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['option'] = {'GPUindex': 1}

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run the algorithm
    nIters = 7
    neach = 50
    print(f'Running {nIters*neach} iterations')
    t = PrintTableMetrics(['Iteration', 'Error'])
    residual_error = np.zeros(nIters)
    for i in range(nIters):
        # Run a single iteration
        astra.algorithm.run(alg_id, neach)
        residual_error[i] = astra.algorithm.get_res_norm(alg_id)
        t.update({'Iteration': i*neach, 'Error': residual_error[i]})

    # Get the result and save
    rec = astra.data3d.get(rec_id)
    np.save(output_folder/f'reconstruction_{residual_error[-1]:.1f}.npy', rec)

    plt.figure(2)
    plt.imshow(rec[:,:,256])
    plt.savefig(output_folder/f'reconstruction_{residual_error[-1]:.1f}.png')
    plt.close()

    plt.figure(4)
    plt.plot(residual_error)
    plt.savefig(output_folder/f'residual_error.png')
    plt.close()

    out_dir = output_folder/Path(f'./slices_{proj_data.shape[1]}')
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