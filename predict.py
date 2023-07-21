from cog import BasePredictor, Input, Path
import torch

class Predictor(BasePredictor):
    def setup(self):
        self.model = torch.load("./models//4x_Faces_04_N_180000_G.pth")

    def predict(self,
          image: Path = Input(description="input image")
    ) -> Path:
        return (image)