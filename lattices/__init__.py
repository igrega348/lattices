from .data import Lattice
from .data import Catalogue
from .utils import abaqus
from .utils import elasticity_func

__all__ = [
    'Catalogue',
    'Lattice',
    'abaqus',
    'elasticity_func'
]

try:
    from .utils import plotting
    __all__.append('plotting')
except ImportError:
    pass

classes = __all__