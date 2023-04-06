import time
import math
import re
import sys
import os
import argparse

import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from face_detection import RetinaFace
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
# matplotlib.use('TkAgg')

from model import SixDRepNet
import utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    # 尝试使用video作为应用源
    parser.add_argument("--video", type=str, default=None,
                        help="Path of video to process i.e. /path/to/vid.mp4")
    parser.add_argument('--cam',
                        dest='cam_id', help='Camera device id to use [0]',
                        default=None, type=int) # 此处default在原文件中是0
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    cam = args.cam_id if args.cam_id is not None else args.video
    if(cam is None):
        print('Camera or video not specified as argument, selecting default camera node (0) as input...')
        cam = 0
    snapshot_path = args.snapshot
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    print('Loading data.')

    detector = RetinaFace(gpu_id=gpu)

    # Load snapshot
    # 由于我保存的快照格式和原来的不一样，所以下面的代码对我不太适用
    # saved_state_dict = torch.load(os.path.join(
    #     snapshot_path), map_location='cpu')

    # if 'model_state_dict' in saved_state_dict:
    #     model.load_state_dict(saved_state_dict['model_state_dict'])
    # else:
    #     model.load_state_dict(saved_state_dict)

    model = torch.load("/home/ubuntu/work_space/6DRepNet_Experiment/sixdrepnet/output/snapshots/SixDRepNet_1680038195_bs100_Pose_300W_LP_GeodesicLoss_Convnext/_epoch_30.pth")
    model.cuda(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    cap = cv2.VideoCapture(cam)

    # 以下代码用于保存处理后的视频文件
    # 获取视频的FPS、宽度和高度
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建输出视频的VideoWriter对象
    out = cv2.VideoWriter('../../datasets/test/output_video_new.mp4', fourcc, fps, (width, height))

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    frame_count = 0
    time_consume = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            frame_count += 1

            faces = detector(frame)

            for box, landmarks, score in faces:

                # Print the location of each face in this image
                if score < .95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)

                img = torch.Tensor(img[None, :]).cuda(gpu)

                c = cv2.waitKey(1)
                if c == 27:
                    break

                start = time.time()
                R_pred = model(img)
                end = time.time()
                time_consume += (end - start)*1000.
                print('Head pose estimation: %2f ms' % ((end - start)*1000.))
                print("Processed frame per second: %2f fps" % (1000. / (time_consume / frame_count)))

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                #utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
                utils.plot_pose_cube(frame,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
                    x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)

            out.write(frame)
            cv2.imshow("Demo", frame)
            cv2.waitKey(5)