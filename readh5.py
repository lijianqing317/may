import datetime
import os
import h5py
import numpy as np
with h5py.File('./zys/encoder_model.h5') as f:
 print f.keys()