from typing import Callable, Union, Iterable, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import torch
import yaml
from functools import lru_cache

from .fields import DisplacementField

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

    def __init__(self, phi_x: np.ndarray, dx: float) -> None:
        super().__init__()
        assert phi_x.ndim == 3
        self.phi_x = phi_x
        self.dx = dx


    def displacement(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, i: int, **kwargs):
        ix = np.floor(x/self.dx).astype(int) - 1
        iy = np.floor(y/self.dx).astype(int) - 1
        iz = np.floor(z/self.dx).astype(int) - 1
        u = x/self.dx - ix + 1
        v = y/self.dx - iy + 1
        w = z/self.dx - iz + 1
        T = np.zeros_like(x)
        for l in range(4):
            for m in range(4):
                for n in range(4):
                    T += self.bspline(u, l) * self.bspline(v, m) * self.bspline(w, n) * self.phi_x[ix+l, iy+m, iz+n]
        return T