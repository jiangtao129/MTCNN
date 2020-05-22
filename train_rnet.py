import nets
import os
import train

if __name__ == '__main__':
    param_path = r"param/r_net.pth"
    data_path = r"D:\DataSet\MTCNN\24"
    if not os.path.exists("param"):
        os.makedirs("param")
    net = nets.RNet()
    t = train.Trainer(net, param_path, data_path)
    t.train(0.001)
