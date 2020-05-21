import os
import numpy as np
import tool
from PIL import Image
import traceback

img_dir = r"E:/DataSet/MTCNN/img_celeba"
anno_src = r"E:/DataSet/MTCNN/list_bbox_celeba.txt"
anno_landmarks_src = r"E:/DataSet/MTCNN/list_landmarks_celeba.txt"
save_dir = r"E:/DataSet/MTCNN/landmaks"
# 为随机数种子做准备，使正样本，部分样本，负样本的比例为1：1：3
float_num = [0.1, 0.5, 0.5, 0.95, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]


def gen_sample(face_size, stop_value):
    # 创建保存样本的目录
    positive_img_dir = os.path.join(save_dir, str(face_size), "positive")
    negative_img_dir = os.path.join(save_dir, str(face_size), "negative")
    part_img_dir = os.path.join(save_dir, str(face_size), "part")
    for dir_path in [positive_img_dir, negative_img_dir, part_img_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 创建保存标签的文件，并打开文件
    anno_positive_filename = os.path.join(save_dir, str(face_size), "positive.txt")
    anno_negative_filename = os.path.join(save_dir, str(face_size), "negative.txt")
    anno_part_filename = os.path.join(save_dir, str(face_size), "part.txt")
    try:
        anno_positive_file = open(anno_positive_filename, 'w')
        anno_negative_file = open(anno_negative_filename, 'w')
        anno_part_file = open(anno_part_filename, 'w')

        # 样本计数
        positive_count = 0
        negative_count = 0
        part_count = 0

        # 按行读取5个标记点的标签文件，返回一个列表
        landmarks_list = open(anno_landmarks_src).readlines()

        # 开打人脸框的标签，循环读取每一行
        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue

            # 分割标签，记录图片名和坐标点
            landmarks = landmarks_list[i].split()
            strs = line.split()
            img_name = strs[0].strip()
            print(img_name)
            img = Image.open(os.path.join(img_dir, img_name))
            img_w, img_h = img.size
            x1 = int(strs[1].strip())
            y1 = int(strs[2].strip())
            w = int(strs[3].strip())
            h = int(strs[4].strip()) * 0.9
            x2 = x1 + w
            y2 = y1 + h

            # 记录5个关键点的坐标
            px1 = int(landmarks[1].strip())
            py1 = int(landmarks[2].strip())
            px2 = int(landmarks[3].strip())
            py2 = int(landmarks[4].strip())
            px3 = int(landmarks[5].strip())
            py3 = int(landmarks[6].strip())
            px4 = int(landmarks[7].strip())
            py4 = int(landmarks[8].strip())
            px5 = int(landmarks[9].strip())
            py5 = int(landmarks[10].strip())

            # 判断坐标是否符合要求
            if max(w, h) < 40 or x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                continue
            box = [x1, y1, x2, y2]

            # 求出中心点和边长，偏移中心点和边长得到样本，每张图偏移5次
            cx = x1 + w / 2
            cy = y1 + h / 2
            max_side = max(w, h)
            for _ in range(5):
                seed = float_num[np.random.randint(0, len(float_num))]
                _max_side = max_side + np.random.randint(int(-max_side * seed), int(max_side * seed))
                _cx = cx + np.random.randint(int(-cx * seed), int(cx * seed))
                _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))
                _x1 = _cx - _max_side / 2
                _y1 = _cy - _max_side / 2
                _x2 = _x1 + _max_side
                _y2 = _y1 + _max_side
                if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:
                    continue
                # 记录偏移后的坐标
                cbox = [_x1, _y1, _x2, _y2]
                # 计算两个坐标点和5个关键点的偏移率
                offset_x1 = (x1 - _x1) / _max_side
                offset_y1 = (y1 - _y1) / _max_side
                offset_x2 = (x2 - _x2) / _max_side
                offset_y2 = (y2 - _y2) / _max_side

                offset_px1 = (px1 - _x1) / _max_side
                offset_py1 = (py1 - _y1) / _max_side
                offset_px2 = (px2 - _x1) / _max_side
                offset_py2 = (py2 - _y1) / _max_side
                offset_px3 = (px3 - _x1) / _max_side
                offset_py3 = (py3 - _y1) / _max_side
                offset_px4 = (px4 - _x1) / _max_side
                offset_py4 = (py4 - _y1) / _max_side
                offset_px5 = (px5 - _x1) / _max_side
                offset_py5 = (py5 - _y1) / _max_side
                # 根据偏移后的坐标截图图片，并缩放成要训练的大小
                img_crop = img.crop(cbox)
                img_crop = img_crop.resize((face_size, face_size))
                # 对偏移框和真实框做iou, 根据偏离程度划分样本
                iou = tool.iou(box, np.array([cbox]))[0]
                if iou > 0.7:
                    img_crop.save(os.path.join(positive_img_dir, "{0}.jpg".format(positive_count)))
                    anno_positive_file.write(
                        "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(positive_count, 1,
                            offset_x1, offset_y1,offset_x2,offset_y2, offset_px1, offset_py1, offset_px2, offset_py2,
                            offset_px3, offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                    anno_positive_file.flush()
                    positive_count += 1
                elif 0.4 < iou < 0.65:
                    img_crop.save(os.path.join(part_img_dir, "{0}.jpg".format(part_count)))
                    anno_part_file.write(
                        "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(part_count, 2,
                             offset_x1, offset_y1, offset_x2,offset_y2, offset_px1, offset_py1, offset_px2, offset_py2,
                            offset_px3, offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                    anno_part_file.flush()
                    part_count += 1
                elif iou < 0.2:
                    img_crop.save(os.path.join(negative_img_dir, "{0}.jpg".format(negative_count)))
                    anno_negative_file.write("negative/{0}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count))
                    anno_negative_file.flush()
                    negative_count += 1
            count = positive_count + negative_count + part_count
            if count > stop_value:
                break
    except:
        traceback.print_exc()


if __name__ == '__main__':
    gen_sample(12, 50000)
    gen_sample(24, 50000)
    gen_sample(48, 50000)
