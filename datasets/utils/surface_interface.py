from abc import ABC, abstractmethod
from argparse import Namespace
from numpy.typing import NDArray
from numpy import uint8

class SurfaceInterface(ABC):
    surface_name: str = "NotDefined"
    
    @abstractmethod
    def build_surface(self, **kwargs) -> None: ...

    @property
    @abstractmethod
    def as_frame(self) -> NDArray[uint8]: ...

    @abstractmethod
    def update_surface(self, x: int, y: int) -> None: ...

    @staticmethod
    @abstractmethod
    def surf_params_adaptr(args: Namespace) -> dict: ...