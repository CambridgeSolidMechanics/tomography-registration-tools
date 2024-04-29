from typing import Callable, Union, Iterable, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import torch
from functools import lru_cache
import scipy as sp
from .fields import DisplacementField

def load_bspline_params(path: Union[str, Path], units_multiplier: float=1.0) -> Dict:
    out = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    _float = lambda x: units_multiplier*float(x)
    mappers = {
        "GridSize": int, "GridSpacing": _float, "GridOrigin": _float, 
        "Origin": _float, "Spacing": _float, "Size": int, "Index": int,
        "NumberOfParameters": int, "TransformParameters": _float
    }
    for line in lines:
        if line.startswith('//'):
            continue
        elif line.startswith('(') and line.endswith(')'):
            line = line.lstrip('(').rstrip(')')
            fields = line.split()
            key = fields[0]
            if key in mappers:
                out[key] = list(map(mappers[key], fields[1:]))
            else:
                out[key] = ' '.join(fields[1:])
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
            support_outside: bool = False,
            **kwargs
    ) -> None:
        """Set up the B-spline field.

        Args:
            phi_x (Union[torch.Tensor, np.ndarray]): degrees of freedom 
                of the B-spline field in order [dim, nx, ny, nz]
            support_outside (bool, optional): whether to provide support
                for locations outside the control points. Defaults to False.
        """
        super().__init__()
        if 'class' in kwargs:
            assert kwargs['class'] == 'BSplineField'
        assert phi_x.ndim == 4
        if isinstance(phi_x, np.ndarray):
            phi_x = torch.tensor(phi_x, dtype=torch.float32)
        self.phi_x = phi_x
        _,nx,ny,nz = phi_x.shape
        self.grid_size = np.array([nx, ny, nz])
        # provide support for range -1 to 1 along each dimension
        self.spacing = 2 / (self.grid_size - 3)
        self.origin = -1 - self.spacing
        self.support_outside = support_outside

        # These are parameters from the transform file. Sometimes useful for plotting.
        self.paramsFromFile = kwargs 

        # in real coordinates
        if ('GridOrigin' in kwargs) and ('GridSpacing' in kwargs) and ('GridSize' in kwargs):
            self.real_spacing = np.array(kwargs['GridSpacing'])
            self.real_size = np.array(kwargs['GridSize'])
            assert np.allclose(self.real_size, self.grid_size)
            self.real_origin = np.array(kwargs['GridOrigin'])
            # downscale displacements accordingly
            scale_factor = self.spacing / self.real_spacing
            self.phi_x *= scale_factor.reshape(3,1,1,1)

    def __repr__(self) -> str:
        f = self
        return f"BSplineField(phi_x={f.phi_x.shape}, origin={f.origin}, spacing={f.spacing})\nfull support on {f.origin + f.spacing} to {f.origin + f.spacing*(f.grid_size-2)}\n"

    def displacement(
            self, 
            x: Union[torch.Tensor, np.ndarray],
            y: Union[torch.Tensor, np.ndarray],
            z: Union[torch.Tensor, np.ndarray],
            i: int, 
            **kwargs
    ) -> torch.Tensor:
        """Displacement at points x,y,z in the direction i.

        We implement support for locations beyond control points.

        Args:
            x (Union[torch.Tensor, np.ndarray]): x-coordinates. Can be 1d or meshgrid.
            y (Union[torch.Tensor, np.ndarray]): y-coordinates. -"-
            z (Union[torch.Tensor, np.ndarray]): z-coordinates. -"-
            i (int): index of the displacement direction. (x=0, y=1, z=2)

        Returns:
            torch.Tensor: displacement
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32)
        
        dx, dy, dz = self.spacing
        u = (x - self.origin[0] - dx)/dx
        v = (y - self.origin[1] - dy)/dy
        w = (z - self.origin[2] - dz)/dz
        ix = torch.floor(u).long()
        iy = torch.floor(v).long()
        iz = torch.floor(w).long()
        if not self.support_outside:
            ix_nan = (ix < 0) | (ix >= self.grid_size[0]-3)
            iy_nan = (iy < 0) | (iy >= self.grid_size[1]-3)
            iz_nan = (iz < 0) | (iz >= self.grid_size[2]-3)
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
        if not self.support_outside:
            T[ix_nan | iy_nan | iz_nan] = torch.nan
        return T
    
    def get_A_matrix(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Calculate the A-matrix.

        Displacements are then given by
        ..math::
            u(x) = A(x) \\phi_x
        where A(x) is the 2d matrix of B-spline weights evaluated 
        at x and \\phi_x is the 1d vector of B-spline degrees of freedom. 
        The matrix can be used to calculate the displacement at x 
        or to infer weights from displacements.

        Args:
            x (torch.Tensor): x-position. Must be 1d. Shape [npoints]
            y (torch.Tensor): y-position. -"-
            z (torch.Tensor): z-position. -"-

        Returns:
            torch.Tensor: Matrix of shape [npoints, nx*ny*nz]
        """
        assert x.ndim == 1 and y.ndim == 1 and z.ndim == 1
        assert x.shape[0] == y.shape[0] == z.shape[0]
        dx, dy, dz = self.spacing
        nx, ny, nz = self.grid_size
        npoints = x.shape[0]
        u = (x - self.origin[0] - dx)/dx
        v = (y - self.origin[1] - dy)/dy
        w = (z - self.origin[2] - dz)/dz
        ix = torch.floor(u).long()
        iy = torch.floor(v).long()
        iz = torch.floor(w).long()
        u = (u - ix).to(torch.float64)
        v = (v - iy).to(torch.float64)
        w = (w - iz).to(torch.float64)
        ind_x = (ix[:, None] + torch.arange(4, device=ix.device)).repeat_interleave(16, dim=1)
        ind_y = (iy[:, None] + torch.arange(4, device=iy.device)).repeat(1, 4).repeat_interleave(4, dim=1)
        ind_z = (iz[:, None] + torch.arange(4, device=iz.device)).repeat(1, 16)
        out_of_support = (ind_x < 0) | (ind_x >= nx) | (ind_y < 0) | (ind_y >= ny) | (ind_z < 0) | (ind_z >= nz)
        out_of_support = out_of_support.any(dim=1)
        flat_index = np.ravel_multi_index((ind_x, ind_y, ind_z), (nx, ny, nz), mode='clip')
        flat_index = torch.tensor(flat_index, device=x.device, dtype=torch.long)
        weights_x = torch.stack([self.bspline(u, i) for i in range(4)], dim=1)
        weights_y = torch.stack([self.bspline(v, i) for i in range(4)], dim=1)
        weights_z = torch.stack([self.bspline(w, i) for i in range(4)], dim=1)
        weights = torch.einsum('pi,pj,pk->pijk', weights_x, weights_y, weights_z).reshape(npoints, 64)
        del u, v, w, weights_x, weights_y, weights_z, ind_x, ind_y, ind_z, ix, iy, iz
        idx0 = torch.arange(npoints).reshape(-1, 1).repeat(1, 64).reshape(1, -1)
        flat_index = flat_index.reshape(1, -1)
        weights[out_of_support] = torch.nan
        weights = weights.flatten()
        assert idx0.shape == flat_index.shape
        # A = torch.sparse_coo_tensor(torch.vstack([idx0, flat_index]), weights, size=(npoints, nx*ny*nz), dtype=torch.float64)
        A = sp.sparse.coo_matrix((weights.numpy().flatten(), (idx0.numpy().flatten(), flat_index.numpy().flatten())), shape=(npoints, nx*ny*nz))
        return A
    
    def compute_weights_from_displacement(
            self, 
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            u: torch.Tensor
    ) -> np.ndarray:
        # use double precision for stability
        assert x.ndim==1 and y.ndim==1 and z.ndim==1 and u.ndim==1
        # A = self.get_A_matrix(x,y,z).double()
        # u = u.double()
        # weights = torch.linalg.lstsq(A, u, rcond=None).solution.float()

        A = self.get_A_matrix(x,y,z)
        weights = sp.sparse.linalg.lsqr(A, u.numpy().flatten())[0]
        weights = torch.Tensor(weights)
        # reshape to 3d. However, this  still only gives 
        # the weights for one component of the displacement
        return weights.reshape(1, *self.grid_size)
    
    def real_displacement(
            self,
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            i: int,
            **kwargs
    ) -> torch.Tensor:
        """Return displacements in real coordinates.

        Unlike the displacement method (which assumes volumes go between -1 and 1),
        this takes into account the real extent of the volume from elastix.

        Args:
            x (torch.Tensor): x-coordinates. Can be 1d or meshgrid.
            y (torch.Tensor): y-coordinates. -"-
            z (torch.Tensor): z-coordinates. -"-
            i (int): index of the displacement direction. (x=0, y=1, z=2)

        Returns:
            torch.Tensor: displacement
        """
        # handle 1 element spline using clip
        size = np.clip(self.real_size-1, 1, a_max=None)*self.real_spacing # in physical coords
        slope = self.spacing / self.real_spacing
        intercept = - (1+self.spacing) * (1 + 2*self.real_origin / size)
        return 1/slope[i] * self.displacement(
            x*slope[0] + intercept[0],
            y*slope[1] + intercept[1],
            z*slope[2] + intercept[2],
            i
        )

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
        nx, ny, nz = params['GridSize']
        phi_x = np.array(params['TransformParameters']).reshape(3, nz, ny, nx)
        phi_x = phi_x.swapaxes(1,3)
        return BSplineField(phi_x, **params)
    
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
        
    def __init__(self, phi_x: np.ndarray, support_outside: bool = False) -> None:
        if not isinstance(phi_x, np.ndarray):
            phi_x = np.array(phi_x)
        assert phi_x.ndim == 1
        assert phi_x.shape[0] > 3
        self.phi_x = phi_x
        # provide support over -1 to 1
        self.dx = 2/(len(phi_x)-3)
        self.origin = -1 - self.dx
        self.support_outside = support_outside

    def displacement(self, _t: np.ndarray) -> np.ndarray:
        # # support on -1 to 1
        # _t = 0.5 * (1+_t)*(self.phi_x.shape[0]-3) * self.dx
        assert _t.ndim == 1
        t = _t - self.origin - self.dx
        x = np.zeros_like(t)
        indices = np.floor(t/self.dx).astype(int)        
        if not self.support_outside:
            invalid = (indices < 0) | (indices >= len(self.phi_x)-3)
            valid = ~invalid
            x[invalid] = np.nan
        else:
            valid = np.ones_like(indices, dtype=bool)

        for i in range(4):
            inds_loc = indices[valid] + i
            if self.support_outside:
                inds_loc = np.clip(inds_loc, 0, len(self.phi_x)-1) # support outside the domain
            x[valid] += self.bspline(t[valid]/self.dx - indices[valid], i) * self.phi_x[inds_loc]

        return x