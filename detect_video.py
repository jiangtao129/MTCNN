import tool
import nets
import torch
import numpy as np
from torchvision import transforms
import time
from PIL import Image, ImageDraw
import cv2
import os

"""
由于P网络检测时间较长，因此将P网络检测中取框的for循环，用数组切片形式替代
"""
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class Detector:
    def __init__(self, pnet_path, rnet_path, onet_path):
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
        # print("pnet:", pnet_boxes.shape)
        if pnet_boxes.shape[0] == 0:
            print("P网络为检测到人脸")
            return np.array([])
        end_time = time.time()
        pnet_time = end_time - start_time

        start_time = time.time()
        rnet_boxes = self.rnet_detect(image, pnet_boxes)
        # print("rnet:", rnet_boxes.shape)
        if rnet_boxes.shape[0] == 0:
            print("R网络为检测到人脸")
            return np.array([])
        end_time = time.time()
        rnet_time = end_time - start_time

        start_time = time.time()
        onet_boxes = self.onet_detect(image, rnet_boxes)
        # print("onet:", onet_boxes.shape)
        if onet_boxes.shape[0] == 0:
            print("O网路未检测到人脸")
            return np.array([])
        end_time = time.time()
        onet_time = end_time - start_time

        sum_time = pnet_time + rnet_time + onet_time
        print("time:{}, pnet_time:{}, rnet_time:{}, onet_time:{}".format(sum_time, pnet_time, rnet_time, onet_time))
        return onet_boxes

    def pnet_detect(self, image):
        boxes = []
        w, h = image.size
        min_side = min(w, h)
        scale = 1

        while min_side > 12:
            # 判断images是何种类型
            # if isinstance(image, (np.ndarray, torch.Tensor)):
            #     # img_data = torch.as_tensor(image, device=device)
            #     img_data = torch.as_tensor(image).to(device)
            #     img_data = img_data.permute(2, 0, 1).float()
            # else:
            #     img_data = self.img_transfrom(image).to(device)
            img_data = self.img_transfrom(image).to(device)
            img_data.unsqueeze_(0)
            _cls, _offset = self.pnet(img_data)
            _cls = _cls[0][0].data.cpu()
            _offset = _offset[0].data.cpu()

            # (n,2)
            indexes = torch.nonzero(_cls > 0.6)
            # for循环改进
            # for index in indexes:
            #     boxes.append(self.box(index, _cls[index[0], index[1]], _offset, scale))
            boxes.extend(self.box(indexes, _cls, _offset, scale))

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)
            image = image.resize((_w, _h))
            min_side = min(_w, _h)

        return tool.nms(torch.stack(boxes).numpy(), 0.3)

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
        # 将建议框转换成正方形在从原图上截取并缩放成24*24，以防止人脸变形
        square_boxes = tool.convert_to_square(pnet_boxes)
        for box in square_boxes:
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            img_crop = image.crop([_x1, _y1, _x2, _y2])
            img_crop = img_crop.resize((24, 24))
            img_data = self.img_transfrom(img_crop).to(device)
            img_dataset.append(img_data)

        # (n,1)  (n,4)
        _cls, _offset = self.rnet(torch.stack(img_dataset))
        # 两种取索引的方法numpy.where和torch.nonzero
        # _cls = _cls.cpu().data.numpy()
        # offset = _offset.cpu().data.numpy()
        # idexes, _ = np.where(_cls > 0.6)
        _cls = _cls.data.cpu()
        _offset = _offset.data.cpu()
        indexes = torch.nonzero(_cls > 0.7)[:, 0]
        for index in indexes:
            box = square_boxes[index]
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            side = _x2 - _x1

            offset = _offset[index]
            x1 = int(_x1 + side * offset[0])
            y1 = int(_y1 + side * offset[1])
            x2 = int(_x2 + side * offset[2])
            y2 = int(_y2 + side * offset[3])
            cls = _cls[index][0]
            boxes.append([x1, y1, x2, y2, cls])

        return tool.nms(np.array(boxes), 0.3)

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
        _cls = _cls.data.cpu()
        _offset = _offset.data.cpu()
        _point = _point.data.cpu()
        indexes = torch.nonzero(_cls > 0.96)[:, 0]
        for index in indexes:
            box = square_boxes[index]
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            side = _x2 - _x1

            offset = _offset[index]
            x1 = int(_x1 + side * offset[0])
            y1 = int(_y1 + side * offset[1])
            x2 = int(_x2 + side * offset[2])
            y2 = int(_y2 + side * offset[3])
            cls = _cls[index][0]

            point = _point[index]
            px1 = int(_x1 + side * point[0])
            py1 = int(_y1 + side * point[1])
            px2 = int(_x1 + side * point[2])
            py2 = int(_y1 + side * point[3])
            px3 = int(_x1 + side * point[4])
            py3 = int(_y1 + side * point[5])
            px4 = int(_x1 + side * point[6])
            py4 = int(_y1 + side * point[7])
            px5 = int(_x1 + side * point[8])
            py5 = int(_y1 + side * point[9])

            boxes.append([x1, y1, x2, y2, cls, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5])

        return tool.nms(np.array(boxes), 0.3, isMin=True)


if __name__ == '__main__':
    img_name = r"08.jpg"
    img_path = os.path.join("./detect_img", img_name)
    # img = Image.open(img_path)
    img = cv2.imread(img_path)[..., ::-1]
    # BGR转成RGB后会导致内存地址不连续
    # img = np.ascontiguousarray(img, dtype=np.float32)
    img = Image.fromarray(img)
    detector = Detector("param/p_net.pth", "param/r_net.pth", "param/o_net.pth")
    onet_boxes = detector.detect(img)
    img = cv2.imread(img_path)
    for box in onet_boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
        for i in range(5, 15, 2):
            cv2.circle(img, (int(box[i]), int(box[i + 1])), radius=2, color=(255, 255, 0), thickness=-1)
    # cv2.imwrite(os.path.join("./out_img", img_name), img)
    cv2.imshow("img", img)
    cv2.waitKey(0)