import tool
import nets
import torch
import numpy as np
from torchvision import transforms
import time
from PIL import Image, ImageDraw
import cv2
import os
from torchvision.ops.boxes import batched_nms, nms

"""
由于P网络检测时间较长，因此将P网络检测中取框的for循环，用数组切片形式替代
"""
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class Detector:
    def __init__(self, pnet_path, rnet_path, onet_path, softnms=False, thresholds=None, factor=0.709):
        if thresholds is None:
            thresholds = [0.6, 0.6, 0.95]
        self.thresholds = thresholds
        self.factor = factor
        self.softnms = softnms

        self.pnet = nets.PNet().to(device)
        self.rnet = nets.RNet().to(device)
        self.onet = nets.ONet().to(device)
        self.pnet.load_state_dict(torch.load(pnet_path))
        self.rnet.load_state_dict(torch.load(rnet_path))
        self.onet.load_state_dict(torch.load(onet_path))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.img_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect(self, image):
        start_time = time.time()
        pnet_boxes = self.pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            print("P网络为检测到人脸")
            return np.array([])
        end_time = time.time()
        pnet_time = end_time - start_time

        start_time = time.time()
        rnet_boxes = self.rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            print("R网络为检测到人脸")
            return np.array([])
        end_time = time.time()
        rnet_time = end_time - start_time

        start_time = time.time()
        onet_boxes = self.onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            print("O网路未检测到人脸")
            return np.array([])
        end_time = time.time()
        onet_time = end_time - start_time

        sum_time = pnet_time + rnet_time + onet_time
        print("time:{}, pnet_time:{}, rnet_time:{}, onet_time:{}".format(sum_time, pnet_time, rnet_time, onet_time))
        return pnet_boxes, rnet_boxes, onet_boxes

    def pnet_detect(self, image):
        boxes = []
        w, h = image.size
        min_side = min(w, h)
        scale = 1

        #去除第一张
        # scale = 0.7
        # image = image.resize((int(w*scale), int(h*scale)))
        while min_side > 12:
            img_data = self.img_transfrom(image).to(device)
            img_data.unsqueeze_(0)
            _cls, _offset = self.pnet(img_data)
            _cls = _cls[0][0].data.cpu()
            _offset = _offset[0].data.cpu()

            # (n,2)
            indexes = torch.nonzero(_cls > self.thresholds[0])
            # for循环改进
            # for index in indexes:
            #     boxes.append(self.box(index, _cls[index[0], index[1]], _offset, scale))
            boxes.extend(self.box(indexes, _cls, _offset, scale))

            scale *= self.factor
            _w = int(w * scale)
            _h = int(h * scale)
            image = image.resize((_w, _h))
            min_side = min(_w, _h)

        if self.softnms:
            return tool.soft_nms(torch.stack(boxes).numpy(), 0.3)
        # return tool.nms(torch.stack(boxes).numpy(), 0.3)
        boxes = torch.stack(boxes)
        return boxes[nms(boxes[:, :4], boxes[:, 4], 0.3)].numpy()

    def box(self, indexes, cls, offset, scale, stride=2, side_len=12):
        # (n,)
        _x1 = (indexes[:, 1] * stride) / scale
        _y1 = (indexes[:, 0] * stride) / scale
        _x2 = (indexes[:, 1] * stride + side_len) / scale
        _y2 = (indexes[:, 0] * stride + side_len) / scale
        side = _x2 - _x1
        # (4, n)
        offset = offset[:, indexes[:, 0], indexes[:, 1]]
        # (n,)
        x1 = (_x1 + side * offset[0])
        y1 = (_y1 + side * offset[1])
        x2 = (_x2 + side * offset[2])
        y2 = (_y2 + side * offset[3])
        # (n,)
        cls = cls[indexes[:, 0], indexes[:, 1]]
        # (n, 5)
        return torch.stack([x1, y1, x2, y2, cls], dim=1)

    def rnet_detect(self, image, pnet_boxes):
        boxes = []
        img_dataset = []
        # 取出正方形框并转成tensor，方便后面用tensor去索引
        square_boxes = torch.from_numpy(tool.convert_to_square(pnet_boxes))
        for box in square_boxes:
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            # crop裁剪的时候超出原图大小的坐标会自动填充为黑色
            img_crop = image.crop([_x1, _y1, _x2, _y2])
            img_crop = img_crop.resize((24, 24))
            img_data = self.img_transfrom(img_crop).to(device)
            img_dataset.append(img_data)
        # (n,1) (n,4)
        _cls, _offset = self.rnet(torch.stack(img_dataset))

        _cls = _cls.data.cpu()
        _offset = _offset.data.cpu()
        # (14,)
        indexes = torch.nonzero(_cls > self.thresholds[1])[:, 0]

        # (n,5)
        box = square_boxes[indexes]
        # (n,)
        _x1 = box[:, 0]
        _y1 = box[:, 1]
        _x2 = box[:, 2]
        _y2 = box[:, 3]
        side = _x2 - _x1
        # (n,4)
        offset = _offset[indexes]
        # (n,)
        x1 = _x1 + side * offset[:, 0]
        y1 = _y1 + side * offset[:, 1]
        x2 = _x2 + side * offset[:, 2]
        y2 = _y2 + side * offset[:, 3]
        # (n,)
        cls = _cls[indexes][:, 0]
        # np.array([x1, y1, x2, y2, cls]) (5,n)
        boxes.extend(torch.stack([x1, y1, x2, y2, cls], dim=1))
        if len(boxes) == 0:
            return np.array([])

        boxes = torch.stack(boxes)
        return boxes[nms(boxes[:, :4], boxes[:, 4], 0.3)].numpy()

    def onet_detect(self, image, rnet_boxes):
        boxes = []
        img_dataset = []
        square_boxes = tool.convert_to_square(rnet_boxes)
        for box in square_boxes:
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            img_crop = image.crop([_x1, _y1, _x2, _y2])
            img_crop = img_crop.resize((48, 48))
            img_data = self.img_transfrom(img_crop).to(device)
            img_dataset.append(img_data)

        _cls, _offset, _point = self.onet(torch.stack(img_dataset))
        _cls = _cls.data.cpu().numpy()
        _offset = _offset.data.cpu().numpy()
        _point = _point.data.cpu().numpy()
        indexes, _ = np.where(_cls > self.thresholds[2])
        # (n,5)
        box = square_boxes[indexes]
        # (n,)
        _x1 = box[:, 0]
        _y1 = box[:, 1]
        _x2 = box[:, 2]
        _y2 = box[:, 3]
        side = _x2 - _x1
        # (n,4)
        offset = _offset[indexes]
        # (n,)
        x1 = _x1 + side * offset[:, 0]
        y1 = _y1 + side * offset[:, 1]
        x2 = _x2 + side * offset[:, 2]
        y2 = _y2 + side * offset[:, 3]
        # (n,)
        cls = _cls[indexes][:, 0]
        # (n,10)
        point = _point[indexes]
        px1 = _x1 + side * point[:, 0]
        py1 = _y1 + side * point[:, 1]
        px2 = _x1 + side * point[:, 2]
        py2 = _y1 + side * point[:, 3]
        px3 = _x1 + side * point[:, 4]
        py3 = _y1 + side * point[:, 5]
        px4 = _x1 + side * point[:, 6]
        py4 = _y1 + side * point[:, 7]
        px5 = _x1 + side * point[:, 8]
        py5 = _y1 + side * point[:, 9]
        # np.array([x1, y1, x2, y2, cls, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5]) (15,n)
        boxes.extend(np.stack([x1, y1, x2, y2, cls, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5], axis=1))

        if len(boxes) == 0:
            return np.array([])
        return tool.nms(np.stack(boxes), 0.3, isMin=True)


if __name__ == '__main__':
    img_path = r"./data/detect_img/06.jpg"
    img = Image.open(img_path)
    detector = Detector("param/p_net.pth", "param/r_net.pth", "param/o_net.pth")
    pnet_boxes, rnet_boxes, onet_boxes = detector.detect(img)
    img = cv2.imread(img_path)
    for box in onet_boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
        for i in range(5, 15, 2):
            cv2.circle(img, (int(box[i]), int(box[i + 1])), radius=2, color=(255, 255, 0), thickness=-1)
    # cv2.imshow("img", img)
    cv2.waitKey(0)
