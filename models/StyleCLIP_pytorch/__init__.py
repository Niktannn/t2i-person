from .torch_utils import *
from . import torch_utils
from .dnnlib import *
from . import dnnlib
from .training import *
from . import training
import sys
sys.modules['torch_utils'] = torch_utils
sys.modules['dnnlib'] = dnnlib
sys.modules['training'] = training
