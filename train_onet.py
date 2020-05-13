import nets
import os
import train

if __name__ == '__main__':
    param_path = "param_abandon/o_net.pth"
    data_path = r"E:/DataSet/MTCNN/landmaks/48"
    if not os.path.exists("param_abandon"):
        os.makedirs("param_abandon")
    net = nets.ONet()
    t = train.Trainer(net, param_path, data_path)
    t.train(0.001, landmark=True)