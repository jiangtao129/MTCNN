import nets
import os
import train

if __name__ == '__main__':
    param_path = "param_abandon/p_net.pth"
    data_path = r"E:/DataSet/MTCNN/7.5w/12"
    if not os.path.exists("param_abandon"):
        os.makedirs("param_abandon")
    pnet = nets.PNet()
    t = train.Trainer(pnet, param_path, data_path)
    t.train(0.01)