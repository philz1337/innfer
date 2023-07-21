from cog import BasePredictor, Input, Path
import torch

class Predictor(BasePredictor):
    def setup(self):
        self.model = torch.load("./models//4x_Faces_04_N_180000_G.pth")
        

    def predict(self,
          image: Path = Input(description="input image"),
          version: str = Input(
                description='GFPGAN version',
                choices=['8x_NMKD-Faces_160000_G'],
                default='8x_NMKD-Faces_160000_G'),
        scale: float = Input(description='Rescaling factor', default=2),
    ) -> Path:
        return (image)