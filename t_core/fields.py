from typing import Callable, Union, Iterable, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import torch
import yaml
from functools import lru_cache

class Field:
    """General class for deformation field.

    Specific fields can inherit from this class and implement the deformation method.
    """
    def __init__(self) -> None:
        pass

    def deformation(
            self,
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            i: int
    ) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, vol: np.ndarray) -> np.ndarray:
        """Apply the deformation field to volume.

        Args:
            vol (np.ndarray): 3d volume

        Returns:
            np.ndarray: deformed volume
        """
        res = vol.shape
        vol_torch = torch.from_numpy(vol).view(1,1,*res)
        x = torch.linspace(-1, 1, res[0])
        y = torch.linspace(-1, 1, res[1]) 
        z = torch.linspace(-1, 1, res[2]) 
        X,Y,Z = torch.meshgrid(x,y,z, indexing='ij')
        X1 = self.deformation(X,Y,Z,0)
        Y1 = self.deformation(X,Y,Z,1)
        Z1 = self.deformation(X,Y,Z,2)
        grid = torch.stack((Z1,Y1,X1), dim=-1).unsqueeze(0) # pytorch uses order z,y,x
        deformed_volume = torch.nn.functional.grid_sample(vol_torch, grid, align_corners=True)
        vol_b = deformed_volume.squeeze().numpy()
        return vol_b
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save the field to yaml file.

        Requires child classes to implement the to_dict method.
        Args:
            path (Union[str, Path]): path to save the field
        """
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)

class DisplacementField(Field):
    """Class for fields where the deformation can be conveniently expressed as a displacement.

    Specific fields can inherit from this class and implement the displacement method.
    """
    def __init__(self) -> None:
        pass

    def displacement(
            self, 
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            i: int,
            **kwargs
    ):
        raise NotImplementedError
    
    def deformation(
            self,
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            i: int,
            **kwargs
    ) -> torch.Tensor:
        if i == 0:
            return x + self.displacement(x,y,z,i, **kwargs)
        elif i == 1:
            return y + self.displacement(x,y,z,i, **kwargs)
        elif i == 2:
            return z + self.displacement(x,y,z,i, **kwargs)
        else:
            raise ValueError(f"i={i} is not valid")
    
class GaussianDisplacementField(DisplacementField):
    """Localised Gaussian displacement field.
    """
    def __init__(
            self,
            sigma_xyz: Iterable[float],
            A_xyz: Iterable[float],
            r0_xyz: Iterable[float] = (0,0,0),
            **kwargs
    ) -> None:
        """Create a Gaussian deformation field.

        Depending on the values of sigma_xyz and A_xyz, the field can be
        localised to an ellipsoidal region {sigma_x, sigma_y, sigma_z}>0,
        cylindrical region (e.g. {sigma_x, sigma_y}>0, sigma_z=0), or 
        planar region (e.g. {sigma_x, sigma_y}=0, sigma_z>0).
        Args:
            sigma_xyz (Iterable[float]): Length-scales of field in x, y, z
                Any zero value will result in effectively infinite length-scale
            A_xyz (Iterable[float]): Magnitude of field in x, y, z.
            r0_xyz (Iterable[float], optional): Spatial centering of the Gaussian.
                Volume bounds are (-1,1) in all dimensions. Defaults to (0,0,0).
        """
        super().__init__()
        if 'class' in kwargs:
            assert kwargs['class'] == 'GaussianDisplacementField'

        self.sigma_xyz = sigma_xyz
        self.A_xyz = A_xyz
        self.r0_xyz = r0_xyz
        for s,a in zip(sigma_xyz, A_xyz):
            assert s >= 0
            if s>0:
                assert a < s*np.exp(-0.5)
        # note for this type of field, the requirement for compatibility is
        # A_vox < sigma_vox*exp(-0.5) ~ 1.65*sigma_vox

    @lru_cache
    def gaussian_magnitude(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        magnitude = 1.0
        for s, x_i, r_i in zip(self.sigma_xyz, [x,y,z], self.r0_xyz):
            if s > 0:
                if isinstance(magnitude, float):
                    magnitude = torch.exp(-0.5*((x_i - r_i)**2/s**2))
                else:
                    magnitude *= torch.exp(-0.5*((x_i - r_i)**2/s**2))
        return magnitude

    def displacement(
            self, 
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            i: int,
    ) -> torch.Tensor:
        gaussian_magnitude = self.gaussian_magnitude(x,y,z)
        return self.A_xyz[i]*gaussian_magnitude
    
    def to_dict(self) -> Dict:
        return {
            'class': 'GaussianDisplacementField',
            'sigma_xyz': self.sigma_xyz,
            'A_xyz': self.A_xyz,
            'r0_xyz': self.r0_xyz
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'GaussianDisplacementField':
        return GaussianDisplacementField(**data)
    
class UniformDisplacementField(DisplacementField):
    def __init__(self, u_xyz: Iterable[float], **kwargs) -> None:
        """Uniform rigid body displacement field.

        Args:
            u_xyz (Iterable[float]): x,y,z displacements in normalized coordinates.
        """
        super().__init__()
        if 'class' in kwargs:
            assert kwargs['class'] == 'UniformDisplacementField'
        self.u_xyz = u_xyz
    
    def displacement(
            self, 
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            i: int,
    ) -> torch.Tensor:
        return self.u_xyz[i] * torch.ones_like(x)
    
    def to_dict(self) -> Dict:
        return {
            'class': 'UniformDisplacementField',
            'u_xyz': self.u_xyz
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'UniformDisplacementField':
        return UniformDisplacementField(**data)
    
class UniformStrainDisplacementField(DisplacementField):
    def __init__(self, eps_xyz: Iterable[float], **kwargs) -> None:
        """Create a uniform strain displacement field.

        Args:
            eps_xyz (Iterable[float]): strains eps_xx, eps_yy, eps_zz
        """
        super().__init__()
        if 'class' in kwargs:
            assert kwargs['class'] == 'UniformStrainDisplacementField'
        self.eps_xyz = eps_xyz
        if 'offset' in kwargs:
            self.offset_xyz = kwargs['offset_xyz']
            assert len(self.offset_xyz)==3
        else:
            self.offset_xyz = [0.0, 0.0, 0.0]
    
    def displacement(
            self, 
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            i: int,
    ) -> torch.Tensor:
        if i == 0:
            return self.eps_xyz[i]*(x-self.offset_xyz[i])
        elif i == 1:
            return self.eps_xyz[i]*(y-self.offset_xyz[i])
        elif i == 2:
            return self.eps_xyz[i]*(z-self.offset_xyz[i])
    
    def to_dict(self) -> Dict:
        return {
            'class': 'UniformStrainDisplacementField',
            'eps_xyz': self.eps_xyz
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'UniformStrainDisplacementField':
        return UniformStrainDisplacementField(**data)

    
class AdditiveFieldArray(DisplacementField):
    """General field which is the sum of multiple displacement fields.
    """
    def __init__(self, fields: Iterable[DisplacementField]) -> None:
        """Create a field which is the sum of multiple displacement fields.

        Args:
            fields (Iterable[DisplacementField]): an iterable of displacement fields.
                Each of the displacement fields should implement the displacement method.
        """
        self.fields = fields
    
    def displacement(
            self, 
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            i: int,
            **kwargs
    ) -> torch.Tensor:
        u = torch.stack([f.displacement(x,y,z,i, **kwargs) for f in self.fields], dim=-1)
        return u.sum(dim=-1)
        
    def to_dict(self) -> Dict:
        return {
            'class': 'AdditiveFieldArray',
            'fields': [f.to_dict() for f in self.fields]
        }
    
    @staticmethod 
    def from_dict(data: Dict) -> 'AdditiveFieldArray':
        assert 'class' in data
        assert data['class'] == 'AdditiveFieldArray'
        fields = []
        for f in data['fields']:
            cls_name = f['class']
            if cls_name == 'GaussianDisplacementField':
                fields.append(GaussianDisplacementField.from_dict(f))
            elif cls_name == 'UniformDisplacementField':
                fields.append(UniformDisplacementField.from_dict(f))
            elif cls_name == 'UniformStrainDisplacementField':
                fields.append(UniformStrainDisplacementField.from_dict(f))
            # any other fields which can be composed by addition can be added here
            else:
                raise ValueError(f"Unknown class {cls_name}")
        return AdditiveFieldArray(fields)