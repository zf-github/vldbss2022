import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_layers):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_layers, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data, 0, 0.1)
                nn.init.uniform_(m.weight.data,-0.0001, 0.0001)
                m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
