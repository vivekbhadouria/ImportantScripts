#!/usr/bin/env python
#python tools/demo_video_args.py --net output/marker/train/zf_faster_rcnn_basketball_iter_1000.caffemodel --prototxt models/basketball/test.prototxt 

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

"""
Vivek
This scripts has been tested with OpenCV 2.9. It will throw an error with OpenCV 3
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


# Vivek : DEFINE YOUR CLASS HERE
# n02974003 : Tyre; n04037443 : Car
CLASSES = ('__background__',
           'n02974003','n04037443')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    # Vivek. Change the name of the video
    # print os.getcwd()
    cap = cv2.VideoCapture('TireWorks.mp4')

    count = 1

    while(cap.isOpened()):
        ret, frame = cap.read()

        if count == 1:
		(h, w) = frame.shape[:2]
		zeros = np.zeros((h, w), dtype="uint8")
		count = 0
    
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret:
           timer = Timer()
           timer.tic()
           scores, boxes = im_detect(net, frame)
           timer.toc()
           print ('Detection took {:.3f}s for '
                  '{:d} object proposals').format(timer.total_time, boxes.shape[0])

           CONF_THRESH = 0.8
           NMS_THRESH = 0.3
           for cls_ind, cls in enumerate(CLASSES[1:]):
               cls_ind += 1 # because we skipped background
               cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
               cls_scores = scores[:, cls_ind]
               dets = np.hstack((cls_boxes,
                                 cls_scores[:, np.newaxis])).astype(np.float32)
               keep = nms(dets, NMS_THRESH)
               dets = dets[keep, :]

               inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
               for i in inds:
                   bbox = dets[i, :4]
                   cv2.putText(frame, cls, (bbox[0], int(bbox[3] + 25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.CV_AA)
                   cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255,0), 3)

           (B, G, R) = cv2.split(frame)
           R = cv2.merge([zeros, zeros, R])
           G = cv2.merge([zeros, G, zeros])
           B = cv2.merge([B, zeros, zeros])

           output = np.zeros((h, w, 3), dtype="uint8")
           output = frame

           cv2.imshow('Deep Learning Demonstration', frame)

           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
        else:
           break

    cap.release()
    #vw.release()
    #del vw
    cv2.destroyAllWindows()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='caffe model', default='vgg16')
    parser.add_argument('--prototxt', dest='prototxt', help='Prototxt', default='models/FACEACT/ZF/faster_rcnn_end2end/test.prototxt')
    parser.add_argument('--videoname', dest='videoname', help='Video Name', default='F1IndianGrandPrix.mp4')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    net = caffe.Net(args.prototxt, args.demo_net, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(args.demo_net)
 
    demo(net, args.videoname)
