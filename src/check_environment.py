import platform
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers

print("Python:", sys.version)
print("Platform:", platform.platform())
print("Architecture:", platform.machine())
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    x = torch.randn(100, 100, device="mps")
    y = x @ x.T
    print("MPS computation successful:", y.shape)