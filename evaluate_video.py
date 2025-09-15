import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import imageio
import matplotlib.pyplot as plt

from network import RAFTGMA
from utils import flow_viz
from utils.utils import InputPadder
import os


DEVICE = 'cuda'


def load_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, flow_dir):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    imageio.imwrite(os.path.join(flow_dir, 'flo.png'), flo)
    print(f"Saving optical flow visualisation at {os.path.join(flow_dir, 'flo.png')}")


def normalize(x):
    return x / (x.max() - x.min())


def demo(args):
    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model))
    print(f"Loaded checkpoint at {args.model}")

    model = model.module
    model.to(DEVICE)
    model.eval()

    flow_dir = os.path.join(args.path, args.model_name)
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    with torch.no_grad():
        
        video_extensions = ('.avi', '.mp4')
        video_files = [f for f in os.listdir(args.path) 
                    if f.lower().endswith(video_extensions)]
        video_files = [os.path.join(args.path, f) for f in video_files]
        
        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)

            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_file}")

            # Read the first frame
            ret, prev_frame = cap.read()

            while True:
                # Read the next frame
                ret, next_frame = cap.read()
                if not ret:
                    break

                
                image1 = load_frame(prev_frame)
                image2 = load_frame(next_frame)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)
                print(f"Estimating optical flow...")

                viz(image1, flow_up, flow_dir)
                
                # Move to the next step: shift frames
                prev_frame = next_frame

            cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    demo(args)
