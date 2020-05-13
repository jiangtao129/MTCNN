import nets
import os
import train

if __name__ == '__main__':
    param_path = "param_abandon/r_net.pth"
    data_path = r"E:/DataSet/MTCNN/7.5w/24"
    if not os.path.exists("param_abandon"):
        os.makedirs("param_abandon")
    net = nets.RNet()
    t = train.Trainer(net, param_path, data_path)
    t.train(0.001)
