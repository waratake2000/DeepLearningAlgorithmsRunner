import torch.nn as nn
import torch.nn.functional as F

class FaceKeypointModel(nn.Module):
    def __init__(self):
        super(FaceKeypointModel, self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=5)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3)

        self.fc1 = nn.Linear(128,18)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        bs,_,_,_ = x.shape
        x = F.adaptive_avg_pool2d(x,1).reshape(bs,-1)
        out = self.fc1(x)

        return out
