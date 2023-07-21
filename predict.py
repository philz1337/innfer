import subprocess
import os
from cog import BasePredictor, Input, Path
import torch

class Predictor(BasePredictor):
    def setup(self):
        self.model = torch.load("./models/8x_NMKD-Faces_160000_G.pth")

    def predict(
        self,
        image: Path = Input(description="input image"),
        version: str = Input(
            description='GFPGAN version',
            choices=['8x_NMKD-Faces_160000_G'],
            default='8x_NMKD-Faces_160000_G'
        ),
        scale: float = Input(description='Rescaling factor', default=2)
    ) -> Path:
        output_path = "/tmp/seed-1.png"
        model_path = f"./models/{version}.pth"

        command = f"python run.py -m {model_path} -o {output_path} -scale {scale}"
        process = subprocess.Popen(command, shell=True)

        while process.poll() is None:
            continue

        if os.path.exists(output_path):
            return Path(output_path)
        else:
            raise FileNotFoundError("No file found.")
