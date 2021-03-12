import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms

class Preproc(nn.Module):

    def __init__(self, agent_params, height, width):
        super(Preproc, self).__init__()
        
        self.agent_params = agent_params
        self.height = height
        self.width = width

        
    def forward(self, x):

        if x.dim() > 3:
            x = x[0]
            
        x = x.float()
     
        if self.agent_params["use_rgb_for_raw_state"]:

            # See https://groups.google.com/forum/#!topic/arcade-learning-environment/JHKeYxTzxvo
            # Weirdly, I still get a slight discrepancy versus the values returned by ale.getScreenGrayscale()
            x = np.dot(x[...,:3], [0.299, 0.587, 0.114])
            x = torch.from_numpy(x)
            x = x.float()

        x = x.reshape(1, 210, 160)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.width, self.height), interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ])

        x = transform(x)

        x = x[0]

        return x
