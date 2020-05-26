from PIL import Image
import cv2
import os
from fast_detect import Detector

# from detect import Detector

# from detect import Detector
# param_rectify是矫正了CeleA数据标签的训练参数

param_path = "param_rectify"
p_net, r_net, o_net = [os.path.join(param_path, "p_net.pth"), os.path.join(param_path, "r_net.pth"),
                       os.path.join(param_path, "o_net.pth")]
img_name = r"06.jpg"
detect_img = "./data/detect_img"
out_img = "./data/out_img"
os.makedirs("./data/out_img", exist_ok=True)

if __name__ == '__main__':
    img_path = os.path.join(detect_img, img_name)
    img = Image.open(img_path)
    detector = Detector(p_net, r_net, o_net, softnms=False, thresholds=[0.6, 0.7, 0.95])
    detect_boxes = detector.detect(img)

    if len(detect_boxes) == 0:
        onet_boxes = detect_boxes
    else:
        pnet_boxes, rnet_boxes, onet_boxes = detect_boxes
        print("pnet:", pnet_boxes.shape)
        print("rnet:", rnet_boxes.shape)
        print("onet:", onet_boxes.shape)
    img = cv2.imread(img_path)
    if onet_boxes.shape[0] != 0:
        for box in onet_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            for i in range(5, 15, 2):
                cv2.circle(img, (int(box[i]), int(box[i + 1])), radius=1, color=(255, 255, 0), thickness=-1)
    # cv2.imwrite(os.path.join(out_img, img_name), img)
    # cv2.imshow("img", img)
    cv2.waitKey(0)
