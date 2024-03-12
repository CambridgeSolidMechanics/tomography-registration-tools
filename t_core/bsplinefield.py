from typing import Callable, Union, Iterable, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import torch
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
    """Cubic B-spline 3d field.
    """
    
    @staticmethod
    @lru_cache
    def bspline(u: torch.Tensor, i: int) -> torch.Tensor:
        """B-spline functions.

        Using lru_cache to speed up the computation if 
        the same u and i are used multiple times.

        Args:
            u (torch.Tensor): coordinate in domain
            i (int): index of the B-spline. One of {0,1,2,3}

        Returns:
            torch.Tensor: B-spline weight
        """
        if i == 0:
            return (1 - u)**3 / 6
        elif i == 1:
            return (3*u**3 - 6*u**2 + 4) / 6
        elif i == 2:
            return (-3*u**3 + 3*u**2 + 3*u + 1) / 6
        elif i == 3:
            return u**3 / 6

    def __init__(
            self, 
            phi_x: Union[torch.Tensor, np.ndarray], 
            origin=(-1,-1,-1), 
            spacing=(1,1,1),
            **kwargs
    ) -> None:
        """Set up the B-spline field.

        Args:
            phi_x (Union[torch.Tensor, np.ndarray]): degrees of freedom 
                of the B-spline field in order [dim, nx, ny, nz]
            origin (tuple, optional): coordinates of the first control point. 
                Defaults to (-1,-1,-1).
            spacing (tuple, optional): spacing between control points 
                along each dimension. Defaults to (1,1,1).
        """
        super().__init__()
        if 'class' in kwargs:
            assert kwargs['class'] == 'BSplineField'
        assert phi_x.ndim == 4
        if isinstance(phi_x, np.ndarray):
            phi_x = torch.tensor(phi_x, dtype=torch.float32)
        self.phi_x = phi_x
        _,nx,ny,nz = phi_x.shape
        self.grid_size = (nx, ny, nz)
        self.origin = origin
        self.spacing = spacing

    def __repr__(self) -> str:
        f = self
        return f"BSplineField(phi_x={f.phi_x.shape}, origin={f.origin}, spacing={f.spacing})\nfull support on {np.array(f.origin) + np.array(f.spacing)} to {np.array(f.origin) + np.array(f.spacing)*(np.array(f.grid_size)-2)}\n"

    def displacement(
            self, 
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, 
            i: int, 
            **kwargs
    ) -> torch.Tensor:
        """Displacement at points x,y,z in the direction i.

        We implement support for locations beyond control points.

        Args:
            x (torch.Tensor): x-coordinates. Can be 1d or meshgrid.
            y (torch.Tensor): y-coordinates. -"-
            z (torch.Tensor): z-coordinates. -"-
            i (int): index of the displacement direction. (x=0, y=1, z=2)

        Returns:
            torch.Tensor: displacement
        """
        dx, dy, dz = self.spacing
        u = (x - self.origin[0] - dx)/dx
        v = (y - self.origin[1] - dy)/dy
        w = (z - self.origin[2] - dz)/dz
        ix = torch.floor(u).long()
        iy = torch.floor(v).long()
        iz = torch.floor(w).long()
        u = u - ix
        v = v - iy
        w = w - iz
        T = torch.zeros_like(x, dtype=torch.float32)
        for l in range(4):
            ix_loc = torch.clamp(ix + l, 0, self.grid_size[0]-1)
            for m in range(4):
                iy_loc = torch.clamp(iy + m, 0, self.grid_size[1]-1)
                for n in range(4):
                    iz_loc = torch.clamp(iz + n, 0, self.grid_size[2]-1)
                    T += self.bspline(u, l) * self.bspline(v, m) * self.bspline(w, n) * self.phi_x[i, ix_loc, iy_loc, iz_loc]
        return T
    
    @staticmethod
    def from_transform_file(
        path: Union[str, Path], units_multiplier: float = 1.0
    ) -> "BSplineField":
        """Load B-spline field from transform file.

        The transform file is deemed to follow the transformix convention.
        Importantly, the degrees of freedom of the b-splines are saved in
        order: [dim, nz, ny, nx]. We convert to order [dim, nx, ny, nz].

        Args:
            path (Union[str, Path]): path to transform file
            units_multiplier (float, optional): number by which to scale 
                length units - e.g. to go from mm to um, use 1000.
                Defaults to 1.0.

        Returns:
            BSplineField
        """
        params = load_bspline_params(path, units_multiplier)
        nx, ny, nz = params['grid_size']
        phi_x = np.array(params['transform_params']).reshape(3, nz, ny, nx)
        phi_x = phi_x.swapaxes(1,3)
        return BSplineField(phi_x, params['origin'], params['spacing'])
    
    @staticmethod
    def from_dict(d: Dict) -> "BSplineField":
        """Load field from dictionary.

        Our convention is to save b-spline dof in order [dim, nx, ny, nz].

        Args:
            d (Dict): dictionary with field parameters

        Returns:
            BSplineField
        """
        phi_x = np.array(d['phi_x']).reshape(3, *d['grid_size'])
        kwargs = {key:val for key,val in d.items() if key not in ['phi_x']}
        kwargs['phi_x'] = phi_x
        return BSplineField(**kwargs)
    
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