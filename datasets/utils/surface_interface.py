from abc import ABC, abstractmethod

class SurfaceInterface(ABC):
    surface_name: str = "NotDefined"
    
    @abstractmethod
    def build_surface(self, **kwargs): ...

    @property
    @abstractmethod
    def as_frame(self): ...

    @abstractmethod
    def update_surface(self, x: int, y: int): ...