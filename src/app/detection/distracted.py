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
import torchvision.transforms as transforms
from PIL import Image
import cv2


class Distracted:
    def __init__(self, time_treshold=3.0, path=sd.MODEL_PATH.value):
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

        # Variables to store the previous prediction
        self.old_predicition = [None, None, None, None, None, None]
        self.i = 0

    def detect_distraction(self, image):
        self.model.eval()

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (slp.INPUT_SIZE.value, slp.INPUT_SIZE.value), antialias=None
                ),  # resize(input_size)
                # transforms.Normalize(slp.COLOR_MEAN.value, slp.COLOR_STD.value),
            ]
        )

        cv2_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = transform(cv2_img).unsqueeze(0).to(self.device)
        # (
        #     torch.nn.functional.interpolate(
        #         (
        #             (torch.tensor(image) - torch.tensor(slp.COLOR_MEAN.value))
        #             * torch.tensor(slp.COLOR_STD.value)
        #         )
        #         .permute(2, 1, 0)
        #         .unsqueeze(0),
        #         size=(slp.INPUT_SIZE.value, slp.INPUT_SIZE.value),
        #         mode="area",
        #     )
        # .float()
        # .to(self.device)
        # )
        with torch.no_grad():
            logits = self.model(image)
        # print(logits)
        pred = torch.argmax(logits, dim=1)

        # get index of the class
        index = pred.item()
        print(index)
        char_index = "c" + str(index)

        # if the class is not 'c0' (i.e. different from 'safe driving')
        # then the driver is distracted, so we update the state and we check
        # the time passed since the driver is distracted, if it is greater than
        # the treshold, we play the alarm
        if char_index != "c0" and char_index in self.old_predicition:
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

        self.i = (self.i + 1) % len(self.old_predicition)
        self.old_predicition[self.i] = char_index
        return self.state
