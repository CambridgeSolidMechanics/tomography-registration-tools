from typing import Callable, Union, Iterable, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import torch
import yaml
from functools import lru_cache

from .fields import DisplacementField
# %%
class BSplineField:

    @staticmethod
    def bspline(u, i):
        if i == 0:
            return (1 - u)**3 / 6
        elif i == 1:
            return (3*u**3 - 6*u**2 + 4) / 6
        elif i == 2:
            return (-3*u**3 + 3*u**2 + 3*u + 1) / 6
        elif i == 3:
            return u**3 / 6

    def __init__(self, phi_x: np.ndarray, origin=(0,0,0), spacing=(1,1,1)) -> None:
        super().__init__()
        assert phi_x.ndim == 4
        self.phi_x = phi_x
        self.origin = origin
        self.spacing = spacing


    def displacement(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, i: int, **kwargs):
        dx, dy, dz = self.spacing
        u = x/dx
        v = y/dy
        w = z/dz
        ix = np.floor(u).astype(int)
        iy = np.floor(v).astype(int)
        iz = np.floor(w).astype(int)
        u = u - ix
        v = v - iy
        w = w - iz
        T = np.zeros_like(x)
        for l in range(4):
            ix_loc = np.clip(ix + l, 0, self.phi_x.shape[0]-1)
            for m in range(4):
                iy_loc = np.clip(iy + m, 0, self.phi_x.shape[1]-1)
                for n in range(4):
                    iz_loc = np.clip(iz + n, 0, self.phi_x.shape[2]-1)
                    T += self.bspline(u, l) * self.bspline(v, m) * self.bspline(w, n) * self.phi_x[i, ix_loc, iy_loc, iz_loc]
        return T
# %%
class Spline1d:
    @staticmethod
    def bspline(u, i):
        if i == 0:
            return (1 - u)**3 / 6
        elif i == 1:
            return (3*u**3 - 6*u**2 + 4) / 6
        elif i == 2:
            return (-3*u**3 + 3*u**2 + 3*u + 1) / 6
        elif i == 3:
            return u**3 / 6
        
    def __init__(self, phi_x: np.ndarray, dx: float) -> None:
        if not isinstance(phi_x, np.ndarray):
            phi_x = np.array(phi_x)
        assert phi_x.ndim == 1
        self.phi_x = phi_x
        self.dx = dx

        self.matrix = np.array([
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 0, 3, 0],
            [1, 4, 1, 0]
        ])

    def displacement(self, t: np.ndarray) -> np.ndarray:
        assert t.ndim == 1
        # x = np.stack([t**3, t**2, t, np.ones_like(t)], axis=1) 
        # return x @ self.matrix @ self.phi_x
        x = np.zeros_like(t)
        indices = np.floor(t/self.dx).astype(int)        
        print(indices)
        for i in range(4):
            
            inds_loc = indices + i
            inds_loc = np.clip(inds_loc, 0, len(self.phi_x)-1) # support outside the domain
            x += self.bspline(t/self.dx - indices, i) * self.phi_x[inds_loc]
        return x
# %%
def load_bspline_params(path: Union[str, Path]) -> Dict:
    out = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    for line in lines:
        if 'GridSize' in line:
            out['grid_size'] = tuple(map(int, line.strip('()').split()[1:]))
        if 'GridSpacing' in line:
            out["spacing"] = tuple(map(float, line.strip('()').split()[1:]))
        if 'GridOrigin' in line:
            out["origin"] = tuple(map(float, line.strip('()').split()[1:]))
        if '(NumberOfParameters' in line:
            out["num_params"] = int(line.strip('()').split()[-1])
        if '(TransformParameters' in line:
            out["transform_params"] = list(map(float, line.strip('()').split()[1:]))
    return out
# %%
def plot_weights(out):
    axis_order = (2,0,1)
    nx,ny,nz = [out['grid_size'][i] for i in axis_order]
    dx,dy,dz = [out['spacing'][i] for i in axis_order]
    x,y,z = [out['origin'][i] for i in axis_order]
    
    p1 = np.array(out['transform_params']).reshape(3,nx,ny,nz)
    x1 = np.arange(nx)*dx + x
    plt.plot(x1, p1[:,0,0,2])
# %%
