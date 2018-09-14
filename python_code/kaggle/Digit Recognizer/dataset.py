import torch
from torch.utils.data import Dataset as dataset

from utility import get_train_data
from utility import get_test_data


class Dataset(dataset):
    def __init__(self, training):

        self.training = training

        if training is True:
            self.data, self.label = get_train_data()

            self.data = torch.FloatTensor(self.data) / 255
            self.label = torch.LongTensor(self.label)
        else:
            self.data = get_test_data()
            self.data = torch.FloatTensor(self.data)

    def __getitem__(self, index):

        data = self.data[index, :, :, :]

        if self.training is False:
            return data

        else:
            label = self.label[index]

            return data, label

    def __len__(self):

        return self.data.shape[0]


train_ds = Dataset(True)
test_ds = Dataset(False)
