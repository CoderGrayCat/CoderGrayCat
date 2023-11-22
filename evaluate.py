import torch
import shutil
from evaluate_config import *



def xywhn2xyxy(x, w=512, h=512, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def process_data():
    folder = os.path.exists('TruthLabel')
    if not folder:
        os.makedirs('TruthLabel')
    folderlist = os.listdir(label_path)
    for i in folderlist:
        label_path_new = os.path.join(label_path, i)
        with open(label_path_new, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        h, w = img_size
        lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h, 0, 0)  # 反归一化
        for _, x in enumerate(lb):
            class_label = int(x[0])  # class
            # cv2.rectangle(img, (x[1], x[2]), (x[3], x[4]), (0, 255, 0))
            with open('TruthLabel/' + i, 'a') as fw:
                fw.write(str(int(x[0])) + ' ' + str(x[1]) + ' ' + str(x[2]) + ' ' + str(x[3]) + ' ' + str(x[4]) + '\n')
    folder = os.path.exists('PredictLabel')
    if not folder:
        os.makedirs('PredictLabel')
    folderlist = os.listdir(predict_path)
    for i in folderlist:
        label_path_new = os.path.join(predict_path,i)
        lb=np.load(label_path_new)
        i=i.replace('npy','txt')
        if(len(lb)>0):
            for _, x in enumerate(lb):
                class_label = int(x[0])  # class
                with open('PredictLabel/' + i, 'a') as fw:# 这里需要把confidence放到第二位
                    fw.write(str(int(x[5])) + ' ' + str(x[4])+' '+str(x[0]) + ' ' + str(x[1]) + ' ' + str(x[2]) + ' ' + str(x[3]) + '\n')
        else:
            with open('PredictLabel/' + i, 'a') as fw:  # 这里需要把confidence放到第二位
                fw.write('')
    print("predict and labels process success!")

process_data()
from caculate_mAP import *

if AUTO_DEL:
    shutil.rmtree('PredictLabel')
    shutil.rmtree('TruthLabel')




