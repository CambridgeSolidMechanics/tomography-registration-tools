from typing import Callable, Union, Iterable, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import torch
import yaml
from functools import lru_cache

from .fields import DisplacementField

def load_bspline_params(path: Union[str, Path], units_multiplier: float=1.0) -> Dict:
    out = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    _float = lambda x: units_multiplier*float(x)
    for line in lines:
        if 'GridSize' in line:
            out['grid_size'] = tuple(map(int, line.strip('()').split()[1:]))
        if 'GridSpacing' in line:
            out["spacing"] = tuple(map(_float, line.strip('()').split()[1:]))
        if 'GridOrigin' in line:
            out["origin"] = tuple(map(_float, line.strip('()').split()[1:]))
        if '(NumberOfParameters' in line:
            out["num_params"] = int(line.strip('()').split()[-1])
        if '(TransformParameters' in line:
            out["transform_params"] = list(map(_float, line.strip('()').split()[1:]))
    return out

class BSplineField(DisplacementField):
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

    def __init__(self, phi_x: np.ndarray, origin=(-1,-1,-1), spacing=(1,1,1)) -> None:
        super().__init__()
        assert phi_x.ndim == 4
        self.phi_x = phi_x
        _,nz,ny,nx = phi_x.shape
        self.grid_size = (nx, ny, nz)
        self.origin = origin
        self.spacing = spacing


    def displacement(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, i: int, **kwargs):
        dx, dy, dz = self.spacing
        u = (x - self.origin[0] - dx)/dx
        v = (y - self.origin[1] - dy)/dy
        w = (z - self.origin[2] - dz)/dz
        ix = np.floor(u).astype(int)
        iy = np.floor(v).astype(int)
        iz = np.floor(w).astype(int)
        u = u - ix
        v = v - iy
        w = w - iz
        T = np.zeros_like(x)
        for l in range(4):
            ix_loc = np.clip(ix + l, 0, self.grid_size[0]-1)
            for m in range(4):
                iy_loc = np.clip(iy + m, 0, self.grid_size[1]-1)
                for n in range(4):
                    iz_loc = np.clip(iz + n, 0, self.grid_size[2]-1)
                    T += self.bspline(u, l) * self.bspline(v, m) * self.bspline(w, n) * self.phi_x[i, iz_loc, iy_loc, ix_loc]
        return T
    
    @staticmethod
    def from_transform_file(path: Union[str, Path], units_multiplier: float = 1.0):
        params = load_bspline_params(path, units_multiplier)
        nx, ny, nz = params['grid_size']
        phi_x = np.array(params['transform_params']).reshape(3, nz, ny, nx)
        return BSplineField(phi_x, params['origin'], params['spacing'])
    
    @staticmethod
    def from_dict(d: Dict):
        phi_x = np.array(d['phi_x']).reshape(3, *d['grid_size'])
        return BSplineField(phi_x, d['origin'], d['spacing'])
    
    def to_dict(self) -> Dict:
        return {
            "class": "BSplineField",
            "phi_x": self.phi_x.flatten().tolist(),
            "grid_size": self.grid_size,
            "origin": self.origin,
            "spacing": self.spacing
        }

class _BSplineField1d:
    """1D B-spline field used for prototyping
    """
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
        
    def __init__(self, phi_x: np.ndarray, dx: float, origin: float = -1.0) -> None:
        if not isinstance(phi_x, np.ndarray):
            phi_x = np.array(phi_x)
        assert phi_x.ndim == 1
        self.phi_x = phi_x
        self.dx = dx
        self.origin = origin

    def displacement(self, _t: np.ndarray) -> np.ndarray:
        assert _t.ndim == 1
        t = _t - self.origin - self.dx
        x = np.zeros_like(t)
        indices = np.floor(t/self.dx).astype(int)        
        for i in range(4):
            inds_loc = indices + i
            inds_loc = np.clip(inds_loc, 0, len(self.phi_x)-1) # support outside the domain
            x += self.bspline(t/self.dx - indices, i) * self.phi_x[inds_loc]
        return x