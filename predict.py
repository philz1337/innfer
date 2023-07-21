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
        output_path = "/tmp/seed-1.png"
        model_path = f"./models/{version}.pth"

        # Execute the command in the background
        command = f"python run.py -m {model_path} -o {output_path} -scale {scale}"
        process = subprocess.Popen(command, shell=True)

        # Wait for the process to finish and the file to be created
        max_wait_time = 60  # Maximum 60 seconds of waiting
        start_time = time.time()
        while process.poll() is None:
            if time.time() - start_time > max_wait_time:
                process.terminate()  # Terminate the process if it exceeds the time limit
                raise TimeoutError("The process did not finish on time.")
            time.sleep(1)

        # Check if the file exists
        if os.path.exists(output_path):
            return Path(output_path)
        else:
            raise FileNotFoundError("The file could not be created.")
