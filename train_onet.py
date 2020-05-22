import nets
import os
import train

if __name__ == '__main__':
    param_path = r"param/o_net.pth"
    data_path = r"D:\DataSet\MTCNN\48"
    if not os.path.exists("param"):
        os.makedirs("param")
    net = nets.ONet()
    t = train.Trainer(net, param_path, data_path)
    t.train(0.001, landmark=True)