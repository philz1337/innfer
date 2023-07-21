import subprocess
import os
import time
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
        output_path = "./tmp"
        model_path = f"./models/{version}.pth"

        # Execute the command in the background
        command = f"python run.py -m {model_path} -o {output_path} -scale {scale}"
        subprocess.run(command, shell=True)

        print_folder_structure()

        output_file = output_path + "/raw.png"
        print(f"Output file: {output_file}")

        return Path(output_file)



def print_folder_structure(folder_path="."):
    for root, dirs, files in os.walk(folder_path):
        level = root.replace(folder_path, "").count(os.sep)
        indent = " " * 4 * (level)
        print(f"{indent}Current Folder: {os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")