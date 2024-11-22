import torch
import torch.nn as nn
import torch.nn.functional as F

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc1 = nn.Linear(ft_in, nb_classes)
        self.fc2 = nn.Linear(nb_classes, ft_in)
        self.project = nn.Sequential(
            nn.Linear(ft_in, ft_in),
            nn.Tanh(),
            nn.Linear(ft_in, 1, bias=False)
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        # print(beta.size())
        seq = (beta * z).sum(1)
        ret = self.fc1(seq)
        ret = self.fc2(ret)
        return ret

