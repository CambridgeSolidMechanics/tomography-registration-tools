# %%
import astra
import numpy as np
import pylab
from pathlib import Path
import cv2 as cv
from utils import load_xtekct, load_angles, load_images

def main():
    Lscale = 15
    image_scale_factor = 4
    proj_data, image_indices = load_images('./data', image_scale_factor)
    print(proj_data.shape)
    
    data = load_xtekct('./data')
    print(data.keys())
    xtekct = data['XTekCT']
    print(xtekct.keys())
    angle_map = load_angles('./data')
    
    vol_geom = astra.create_vol_geom(500, 500, 500)

    angle_arr = np.array([angle_map[i] for i in image_indices]) * np.pi/180

    proj_geom = {
        'type':'cone',
        'DetectorSpacingX':xtekct['DetectorPixelSizeX']*image_scale_factor*Lscale, # L
        'DetectorSpacingY':xtekct['DetectorPixelSizeY']*image_scale_factor*Lscale, # L
        'DetectorRowCount':int(xtekct['DetectorPixelsX']/image_scale_factor), # -
        'DetectorColCount':int(xtekct['DetectorPixelsY']/image_scale_factor), # -
        'ProjectionAngles':angle_arr, # - 
        'DistanceOriginSource':xtekct['SrcToObject']*Lscale, # L
        'DistanceOriginDetector':(xtekct['SrcToDetector']-xtekct['SrcToObject'])*Lscale # L
    }

    # Create empty volume
    cube = np.zeros((500,500,500))
  
    # Create projection data from this
    proj_id = astra.create_sino3d_gpu(cube, proj_geom, vol_geom, returnData=False)
    astra.data3d.store(proj_id, proj_data)

    # Display a single projection image
    pylab.gray()
    pylab.figure(1)
    pylab.imshow(proj_data[:,0,:])
    pylab.show()

    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run 150 iterations of the algorithm
    nIters = 10
    neach = 50
    residual_error = np.zeros(nIters)
    for i in range(nIters):
        # Run a single iteration
        astra.algorithm.run(alg_id, neach)
        residual_error[i] = astra.algorithm.get_res_norm(alg_id)
        print(f'Iteration {i*neach}/{nIters*neach}, residual error: {residual_error[i]:.2f}')


    # Get the result and save
    rec = astra.data3d.get(rec_id)
    np.save(f'reconstruction_{residual_error[-1]:.1f}.npy', rec)

    pylab.figure(2)
    pylab.imshow(rec[:,:,256])

    pylab.figure(4)
    pylab.plot(residual_error)
    pylab.show()

    out_dir = Path(f'./slices_{proj_data.shape[1]}')
    out_dir.mkdir(exist_ok=True, parents=True)
    for i in range(0,rec.shape[2],rec.shape[2]//64):
        cv.imwrite(str(out_dir/f'slice_{i:03d}.png'), ((rec[:,:,i]-rec.min()) / (rec.max()-rec.min())*255).astype(np.uint8))

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)


if __name__ == '__main__':
    main()