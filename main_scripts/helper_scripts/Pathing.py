import os
import numpy as np

class Pathing:
    def save_file_np(
        save_dir: str,
        save_file_name: str,
        to_save: np.ndarray
    ) -> None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(
            os.path.join(save_dir, save_file_name),
            to_save
        )
