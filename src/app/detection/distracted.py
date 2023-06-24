import sys
import torch
import pytorch_lightning as pl
import time
import os
import timm

sys.path.append("./")
from src.train.model import LitEfficientNet
from src.train.config import StaticDataset as sd
from src.app.utils.config import Colors
from src.train.config import StaticLearningParameter as slp


class Distracted:
    def __init__(self, time_treshold=5.0, path=sd.MODEL_PATH.value):
        self.time_treshold = time_treshold
        self.state = {
            "start_time": time.perf_counter(),
            "distracted_time": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "color": Colors.GREEN.value,
            "play_alarm": False,
            "class": "c0",
        }

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        efficient_model = timm.create_model(
            slp.MODEL_NAME_0.value,
            pretrained=True,
            num_classes=len(sd.ACTIVITY_MAP.value.items()),
        )

        self.model = LitEfficientNet.load_from_checkpoint(
            model=efficient_model, checkpoint_path=path, map_location=self.device
        )

    def detect_distraction(self, image):
        self.model.eval()

        image = (
            torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        )
        logits = self.model(image)
        pred = torch.argmax(logits, dim=1)

        # get index of the class
        index = pred.item()
        char_index = "c" + str(index)

        # if the class is not 'c0' (i.e. different from 'safe driving')
        # then the driver is distracted, so we update the state and we check
        # the time passed since the driver is distracted, if it is greater than
        # the treshold, we play the alarm
        if char_index != "c0":
            end_time = time.perf_counter()
            self.state["distracted_time"] += end_time - self.state["start_time"]
            self.state["start_time"] = end_time
            self.state["color"] = Colors.RED.value
            if self.state["distracted_time"] > self.time_treshold:
                self.state["play_alarm"] = True
        else:
            self.state["start_time"] = time.perf_counter()
            self.state["distracted_time"] = 0.0
            self.state["color"] = Colors.GREEN.value
            self.state["play_alarm"] = False

        self.state["class"] = char_index
        return self.state
