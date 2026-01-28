"""Dataset loaders (optional imports allowed)."""

from .base import Recording, WindowedDataset, MultiDataset

try:
    from .ajile12 import AJILE12WindowedDataset, load_ajile12_recordings
except ModuleNotFoundError:  # optional dataset file removed
    AJILE12WindowedDataset = None
    load_ajile12_recordings = None

try:
    from .bci_iv import load_bci_iv_recordings
except ModuleNotFoundError:
    load_bci_iv_recordings = None

try:
    from .h1 import load_h1_recordings
except ModuleNotFoundError:
    load_h1_recordings = None

try:
    from .m2 import load_m2_recordings
except ModuleNotFoundError:
    load_m2_recordings = None

try:
    from .mouse import load_mouse_recordings
except ModuleNotFoundError:
    load_mouse_recordings = None

__all__ = [
    "AJILE12WindowedDataset",
    "load_ajile12_recordings",
    "load_bci_iv_recordings",
    "load_h1_recordings",
    "load_m2_recordings",
    "load_mouse_recordings",
    "Recording",
    "WindowedDataset",
    "MultiDataset",
]
