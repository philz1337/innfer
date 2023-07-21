import subprocess
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
        output_path = f"/tmp/seed-1.png"

        command = f"python run.py -m {version}.pth -o {output_path} -scale {scale}"
        subprocess.run(command, shell=True, check=True)
      
        yield Path(output_path)

