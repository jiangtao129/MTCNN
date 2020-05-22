from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import os
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, path):
        super(FaceDataset, self).__init__()
        self.path = path
        self.datasets = []
        self.datasets.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.datasets.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.datasets.extend(open(os.path.join(path, "part.txt")).readlines())

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, item):
        strs = self.datasets[item].strip().split()
        img_name = strs[0]
        cls = torch.tensor([int(strs[1])], dtype=torch.float32)
        strs[2:] = [float(x) for x in strs[2:]]
        offset = torch.tensor(strs[2:6], dtype=torch.float32)
        point = torch.tensor(strs[6:16], dtype=torch.float32)
        img = Image.open(os.path.join(self.path, img_name))
        img_data = torch.tensor((np.array(img) / 255. - 0.5) / 0.5, dtype=torch.float32)
        img_data = img_data.permute(2, 0, 1)

        return img_data, cls, offset, point


if __name__ == '__main__':
    data = FaceDataset(r"E:\DataSet\MTCNN\landmaks\48")
    print(data[0][0].shape)
    print(data[0][1])
    print(data[0][2])
    print(data[0][3])


    print(len(data))
    dataloder = DataLoader(data, batch_size=5, shuffle=True)
    print(len(dataloder))
    for img_data, cls, offset, point in dataloder:
        print(img_data.shape)
        print(cls)
        print(offset)
        print(point)
        break


