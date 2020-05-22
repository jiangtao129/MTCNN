import torch
from torch.utils.data import DataLoader
from data.face_dataset import FaceDataset
import os
from tqdm.autonotebook import tqdm


class Trainer:
    def __init__(self, net, param_path, data_path):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.net = net.to(self.device)
        self.param_path = param_path
        self.datasets = FaceDataset(data_path)
        self.cls_loss_func = torch.nn.BCELoss().to(self.device)
        self.offset_loss_func = torch.nn.MSELoss().to(self.device)
        self.point_loss_func = torch.nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def train(self, stop_value, landmark=False):
        if os.path.exists(self.param_path):
            self.net.load_state_dict(torch.load(self.param_path))
        else:
            print("NO Param")

        dataloader = DataLoader(self.datasets, batch_size=512, shuffle=True, num_workers=4)
        epochs = 0
        while True:
            dataloader = tqdm(dataloader)
            for i, (img_data, _cls, _offset, _point) in enumerate(dataloader):
                img_data = img_data.to(self.device)
                _cls = _cls.to(self.device)
                _offset = _offset.to(self.device)
                _point = _point.to(self.device)

                if landmark:
                    out_cls, out_offset, out_point = self.net(img_data)
                else:
                    out_cls, out_offset = self.net(img_data)
                out_cls = out_cls.view(-1, 1)
                out_offset = out_offset.view(-1, 4)

                # 选取置信度为0，1的正负样本求置信度损失
                cls_mask = torch.lt(_cls, 2)
                cls = torch.masked_select(_cls, cls_mask)
                out_cls = torch.masked_select(out_cls, cls_mask)
                cls_loss = self.cls_loss_func(out_cls, cls)

                # 选取正样本和部分样本求偏移率的损失
                offset_mask = torch.gt(_cls, 0)
                offset = torch.masked_select(_offset, offset_mask)
                out_offset = torch.masked_select(out_offset, offset_mask)
                offset_loss = self.offset_loss_func(out_offset, offset)

                if landmark:
                    point = torch.masked_select(_point, offset_mask)
                    out_point = torch.masked_select(out_point, offset_mask)
                    point_loss = self.point_loss_func(out_point, point)
                    loss = cls_loss + offset_loss + point_loss
                else:
                    loss = cls_loss + offset_loss

                if landmark:
                    dataloader.set_description(
                        "epochs:{}, loss:{:.4f}, cls_loss:{:.4f}, offset_loss:{:.4f}, point_loss:{:.4f}".format(
                            epochs, loss.float(), cls_loss.float(), offset_loss.float(), point_loss.float()))
                    # print("loss:{0:.4f}, cls_loss:{1:.4f}, offset_loss:{2:.4f}, point_loss:{3:.4f}".format(
                    #     loss.float(), cls_loss.float(), offset_loss.float(), point_loss.float()))
                else:
                    dataloader.set_description(
                        "epochs:{}, loss:{:.4f}, cls_loss:{:.4f}, offset_loss:{:.4f}".format(
                            epochs, loss.float(), cls_loss.float(), offset_loss.float()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            torch.save(self.net.state_dict(), self.param_path)
            epochs += 1
            if loss < stop_value:
                break
