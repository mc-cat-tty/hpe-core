from abc import ABC, abstractmethod
from argparse import Namespace


class SurfaceInterface(ABC):
    surface_name: str = "NotDefined"
    
    @abstractmethod
    def build_surface(self, **kwargs): ...

    @property
    @abstractmethod
    def as_frame(self): ...

    @abstractmethod
    def update_surface(self, x: int, y: int): ...

    @staticmethod
    @abstractmethod
    def surf_params_adaptr(args: Namespace): ...