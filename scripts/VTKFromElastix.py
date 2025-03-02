import pyvista as pv
import argparse
import os
import sys
import glob
sys.path.insert(0,'../tomography-registration-tools/')
import matplotlib.pyplot as plt
import t_core.bsplinefield as bsp
import t_core.volume_tools as vt
import t_core.fields as fl
import numpy as np

from tqdm import tqdm
import vtk
import pickle
try:
    from vtkmodules.util.numpy_support import numpy_to_vtk
except:
    from vtk.util.numpy_support import numpy_to_vtk
axLabel = {0: 'x', 1: 'y', 2:'z'}


def writeVTK(XYZ, UXYZ, F, detF, E, underSampRatio, spacing, outFilePath):
    print('Writing VTK file')
    grid = pv.ImageData()

    grid.dimensions = XYZ[0].shape

    grid.origin = tuple(tmp[0][0, 0, 0] for tmp in XYZ)  # The bottom left corner of the data set

    grid.spacing = tuple(s*underSampRatio for s in spacing) #in um
    # Add the data values to the cell data

    grid.point_data['U'] = np.hstack(tuple(u.numpy().flatten(order="F").reshape(-1, 1) for u in UXYZ))
    grid.active_vectors_name = 'U'

    grid.point_data['F'] = np.moveaxis(np.reshape(F, (3, 3, -1), order="F"), -1, 0)
    grid.point_data['E'] = np.moveaxis(np.reshape(E, (3, 3, -1), order="F"), -1, 0)

    grid.point_data['detF'] = detF.flatten(order='F')
    grid.save(outFilePath)
    
def calculateDispEtc(splineField, volMask, underSampRatio):
    splineField.support_outside = False    
    resolution = splineField.paramsFromFile['Size']
    spacing = splineField.paramsFromFile['Spacing']
    origin = splineField.paramsFromFile['Origin']
    
    print('Calculating displacements and def grad')
    
    xyz = [np.arange(res)*sp+o for res, sp, o in zip(resolution, spacing, origin)]
    xyz = [s[underSampRatio//2::underSampRatio] for s in xyz]
    XYZ = np.meshgrid(*xyz, indexing='ij')
    UXYZ = [splineField.real_displacement(*XYZ, i=i) for i in range(3)]
    for u in UXYZ:
        u[~volMask] = np.nan

    F = [[[] for _ in range(3)] for _ in range(3)]
    E = [[[] for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            tmp = np.empty_like(UXYZ[0])
            tmp[:] = np.nan
            F[i][j] = float(i==j) + np.diff(UXYZ[i], append=np.nan, axis=j)/np.diff(xyz[j])[0]

    F = np.array(F)
    E = 0.5*(np.einsum('ji...,jk...->ik...', F, F)  -np.eye(3).reshape(3, 3, 1, 1,1))

    with np.errstate(invalid='ignore'):
        detF = np.linalg.det(np.moveaxis(F, [0, 1], [-2, -1]))
    
    return XYZ, UXYZ, F, detF, E, spacing

def loadFieldAndMask(tpFile, maskFile):
    splineField = bsp.BSplineField.from_transform_file(tpFile, units_multiplier=1000)
    with open(maskFile, 'rb') as f:
        underSampRatio, volMask = pickle.load(f)
    return splineField, underSampRatio, volMask

def calculateAndWriteVTK(tpFilePath, maskFilePath, outFilePath):
    splineField, underSampRatio, volMask = loadFieldAndMask(tpFilePath, maskFilePath)
    XYZ, UXYZ, F, detF, E, spacing = calculateDispEtc(splineField, volMask, underSampRatio)
    writeVTK(XYZ, UXYZ, F, detF, E, underSampRatio, spacing, outFilePath)

if __name__ == '__main__':
    
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Create VTK files containing dispalcements, deformation gradients and Green strains using Elastix transform parameters.')
    
    parser.add_argument('tpFilePath', type=str, metavar='tp', help='Transform Parameters file generated by Elastix.')
 
    parser.add_argument('maskFilePath', type=str, metavar='mask', help='A pickled file of a tuple containing the undersample ratio and an undersampled binary mask of the fixed volume.')

    parser.add_argument('--outFilePath', type=str, metavar='out', help='Path to the output file.', default='out.vtk')

    args = parser.parse_args()
    
    calculateAndWriteVTK(tpFilePath=args.tpFilePath,
         maskFilePath=args.maskFilePath,
         outFilePath=args.outFilePath
         )