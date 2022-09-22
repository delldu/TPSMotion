from torch import nn
import torch
from torchvision import models
import pdb

class KPDetector(nn.Module):
    """
    Predict K*5 keypoints. here K == 10 ?
    """

    def __init__(self, num_tps, **kwargs):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps

        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features # 512
        self.fg_encoder.fc = nn.Linear(num_features, num_tps*5*2) # (512, 100)
        # num_tps = 10 -- K ?
        # kwargs = {'num_channels': 3, 'bg': True, 'multi_mask': True}

        
    def forward(self, image):
        # image.size() -- [1, 3, 256, 256]
        fg_kp = self.fg_encoder(image) # [1, 100]
        bs, _, = fg_kp.shape
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2.0 - 1.0 # fg_kp.size() -- [1, 100], [-1, 1.0]
        out = {'fg_kp': fg_kp.view(bs, self.num_tps*5, -1)}

        # out['fg_kp'].size() -- [1, 50, 2]
        return out
