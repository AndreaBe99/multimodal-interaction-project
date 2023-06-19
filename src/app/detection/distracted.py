import sys
import torch
import pytorch_lightning as pl

sys.path.append("./")
from src.train.model import LitEfficientNet
from src.train.config import StaticDataset as sd

class Distracted():
    
    def __init__(self, path=sd.MODEL_PATH.value) -> None:
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.model = LitEfficientNet.load_from_checkpoint(
            path, 
            map_location=self.device
        )

    def detect_distraction(self, image):
        self.model.eval()
        
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
        logits = self.model(image)
        pred = torch.argmax(logits, dim=1)
        
        # get index of the class
        index = pred.item()
        char_index = 'c' + str(index)
        
        return char_index