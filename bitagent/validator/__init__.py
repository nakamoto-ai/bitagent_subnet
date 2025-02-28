__version__ = "1.0.23"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
from . import reward
from .forward import forward
from .initiation import initiate_validator
