#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import pdb

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg16submission' :('VGG16','vgg16_fast_rcnn_iter_100000.caffemodel'),          
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5, save_path=''):
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

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    if (save_path==''):
        plt.draw()
    else:
        plt.savefig(save_path)
    
def begin_file_write(filename):
    
    file_pointer = open(filename, 'w')
    return file_pointer
    
def write_detection(file_pointer, label, score, bbox):
    
    file_pointer.write(str(label) + "\n")
    file_pointer.write(str(score) + "\n")
    file_pointer.write(str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + "\n")
    
    
def write_segmentation_mask(file_pointer, label, score, bbox, segmentation_pixels, WIDTH=500):
    
    count = 0
    file_pointer.write(str(label) + "\n")
    file_pointer.write(str(score) + "\n")
    for pixel in segmentation_pixels:
        
        #y1,x1,y2,x2 = bbox[0], bbox[1], bbox[2], bbox[3]     
        x1,y1,x2,y2 = bbox[0], bbox[1], bbox[2], bbox[3]     
        pixel_x = pixel%WIDTH
        pixel_y = int(pixel / WIDTH)

        if (pixel_x >= int(x1) and pixel_x <= int(x2) and pixel_y >= int(y1) and pixel_y <= int(y2) ):        
            file_pointer.write(str(pixel) + " ")
            count = count + 1
    
    file_pointer.write("\n")
    print "Wrote", count, "out of", len(segmentation_pixels), "pixels"
    
def close_file_write(file_pointer):
    
    file_pointer.write("0")
    file_pointer.close()
    # close_file
    
def view_segmentation(im, segmentation_pixels, bbox, WIDTH=500):
    
     for pixel in segmentation_pixels:
        
       # pdb.set_trace()
       # y1,x1,y2,x2 = bbox[0], bbox[1], bbox[2], bbox[3]       
        x1,y1,x2,y2 = bbox[0], bbox[1], bbox[2], bbox[3]  
        pixel_x = pixel%WIDTH
        pixel_y = int(pixel / WIDTH)

        if (pixel_x >= int(x1) and pixel_x <= int(x2) and pixel_y >= int(y1) and pixel_y <= int(y2) ):        
            im[pixel_y,pixel_x,:] = [255,255,255]
   

def find_index(matrix, row):
    """If numpy was not so fucking retarded, I could have used a built in function to find out the index of a row in a matrix"""
        
    if (matrix.shape[1] != len(row)):
        return -1
    for r in range(matrix.shape[0]):
        isEqual = True
        
        for c in range(matrix.shape[1]):
            isEqual = isEqual & (int(matrix[r][c]) == int(row[c]) ) # Since Python somehow has floating point values of different precision here
        
        if (isEqual):
            return r
    
    return -1
                
def process_image (net, image_name, im_root, obj_proposals, output_root, segmask_root, NMS_THRESH=0.3, CONF_THRESH=0.6, out_ext = ".bbox"):
    
    # IO   
    im_file = os.path.join(im_root, image_name+'.jpg')   
    im = cv2.imread(im_file)
    segmask_data  = sio.loadmat( os.path.join(segmask_root, image_name+'_segmsk.mat') )
    segmasks = segmask_data['segmasks']
    
    output_file = begin_file_write( os.path.join(output_root, image_name+out_ext) )
    
    #Detection and output
    scores, boxes = im_detect(net, im, obj_proposals)
    
    for cls_ind in range(1,len(CLASSES)): #Ignore background detections

        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        for i in range(dets.shape[0]):
            
            #idx = np.where( np.all(boxes[:, 4*cls_ind:4*(cls_ind + 1)] == np.array(dets[i][0:4]) ) )
            idx = find_index(boxes[:, 4*cls_ind:4*(cls_ind + 1)], dets[i][0:4])            
            segmask_pixels = np.squeeze(segmasks[0,idx])            
            write_segmentation_mask(output_file, cls_ind, dets[i][4], dets[i][0:4], segmask_pixels)
            #write_detection(output_file, cls_ind, dets[i][4], dets[i][0:4])
            
            imtemp = im.copy()
            view_segmentation(imtemp, segmask_pixels, dets[i][0:4])
            vis_detections(imtemp, CLASSES[cls_ind], dets, thresh=CONF_THRESH, save_path=os.path.join(output_root, image_name+'_'+str(idx)+'_'+str(cls_ind)+'_'+str(i)+'_'+'plot.png'))
    
    close_file_write(output_file)

def demo(net, image_name, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                            image_name + '_boxes.mat')
    obj_proposals = sio.loadmat(box_file)['boxes']

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
                                                                    CONF_THRESH)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16submission')
    parser.add_argument('--input_im_root', dest='input_im_root', default='/media/torrvision/catz/Data/VOCdevkit/VOC2012/JPEGImages')
    parser.add_argument('--output_root', dest='output_root', default='/media/torrvision/catz/pascal-bbox/correct')  
    parser.add_argument('--segmask_root', dest='segmask_root', default='/media/torrvision/catz/selective_search_data_own/')                  
    parser.add_argument('--i', dest='i', default=-1, type=int)    
    
    args = parser.parse_args()

    return args
    
def get_filenames_and_proposals(data_file = "voc_2012_trainval.mat"):
    
    data_file = os.path.join(cfg.ROOT_DIR, 'data', 'selective_search_data', data_file)
    data = sio.loadmat(data_file, struct_as_record=True)
    
    boxes = data['boxes']
    image_filenames = data['images']

    return boxes, image_filenames    

def main():
    #Initial setup
    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

 #   if args.cpu_mode:
 #       caffe.set_mode_cpu()
 #   else:
 #       caffe.set_mode_gpu()
 #       caffe.set_device(args.gpu_id)
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    
    print '\n\nLoaded network {:s}'.format(caffemodel)
    
    input_im_root = args.input_im_root
    output_root = args.output_root
    segmask_root = args.segmask_root
    
    #Actual stuff
    #proposals, filenames = get_filenames_and_proposals("voc_2012_trainval.mat")
    filenames_file = os.path.join(cfg.ROOT_DIR, 'data', 'VOC2012', 'ImageSets', 'Segmentation', 'trainval.txt')
    filenames_fp = open(filenames_file, 'r')    
    filenames =  filenames_fp.readlines()
    filenames_fp.close()
    
    i = args.i
    
    start = 0
    end = len(filenames)
    if (i != -1):
        start = i
        end = i+1
    
    for i in range(start, end):
        #imname = filenames[i]
        #imname = [str(''.join(letter)) for letter_array in filenames[i] for letter in letter_array]
        #imname = imname[0]
        i = 1
        imname = filenames[i].strip()

        proposal_file = sio.loadmat( os.path.join(cfg.ROOT_DIR, 'data', 'selective_search_data_own', imname+'_bbox.mat') )        
        prop = proposal_file['boxes']
        
        print i, ":", imname 
        process_image(net, imname, input_im_root, prop, output_root, segmask_root)
        
        #break
    
if __name__ == '__main__':
    main()