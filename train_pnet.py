import nets
import os
import train

if __name__ == '__main__':
    param_path = r"param/p_net.pth"
    data_path = r"D:\DataSet\MTCNN\12"
    if not os.path.exists("param"):
        os.makedirs("param")
    pnet = nets.PNet()
    t = train.Trainer(pnet, param_path, data_path)
    t.train(0.01)