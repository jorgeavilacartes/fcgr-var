from pathlib import Path
from typing import List
import numpy as np

class InputOutputLoader:

    def __call__(self, file_path: Path,):
        "given an image path, return the input-output for the model"
        file_path = str(file_path)
        label = Path(file_path).parent.stem
        npy = np.load(file_path)
        return npy, npy