import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(84, 10)

    def forward(self, inputs):

        outputs = self.conv1(inputs)
        outputs = self.pool(outputs)

        outputs = self.conv2(outputs)
        outputs = self.pool(outputs)

        outputs = outputs.view(-1, 16 * 5 * 5)

        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)

        return outputs