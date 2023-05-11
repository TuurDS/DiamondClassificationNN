import numpy as np
from pyvox.models import Vox
from pyvox.writer import VoxWriter

batchPath = r"D:/neural_network"

test = np.load(f"{batchPath}/test_batch_001_data.npy")
vox = Vox.from_dense(np.unpackbits(test[0], axis=-1).astype(bool))
VoxWriter("test.vox", vox).write()