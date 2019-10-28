#*************************************************************************
# Copyright (C) 2018 Kaan Kara - Systems Group, ETH Zurich

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#*************************************************************************

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
import cv2
from os import listdir
from os.path import isfile, join
from random import shuffle
import xml.etree.ElementTree

import numpy as np

FLAGS = None

def bb_intersection_over_union(boxA, boxB):
  # from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA['xmin'], boxB['xmin'])
  yA = max(boxA['ymin'], boxB['ymin'])
  xB = min(boxA['xmax'], boxB['xmax'])
  yB = min(boxA['ymax'], boxB['ymax'])
 
  # compute the area of intersection rectangle
  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
  # compute the area of both the prediction and ground-truth rectangles
  boxAArea = (boxA['xmax'] - boxA['xmin'] + 1) * (boxA['ymax'] - boxA['ymin'] + 1)
  boxBArea = (boxB['xmax'] - boxB['xmin'] + 1) * (boxB['ymax'] - boxB['ymin'] + 1)
 
  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  if (float(boxAArea + boxBArea - interArea) != 0):
    iou = interArea / float(boxAArea + boxBArea - interArea)
  else:
    print('--------------------------------')
    print('!!!! float division by zero')
    print('boxA = [' + str(boxA['xmin']) + ',' + str(boxA['xmax']) + ',' + str(boxA['ymin']) + ',' + str(boxA['ymax']) + ']')
    print('boxB = [' + str(boxB['xmin']) + ',' + str(boxB['xmax']) + ',' + str(boxB['ymin']) + ',' + str(boxB['ymax']) + ']')
    print('interArea = ' + str(interArea))
    print('boxAArea = ' + str(boxAArea))
    print('boxBArea = ' + str(boxBArea))
    print('--------------------------------')
    iou = 0

  # return the intersection over union value
  return iou

def display():

  images = []
  metas = []

  for file in listdir(FLAGS.image_dir):
      if '.jpg' in file:
        images.append(file)
      if '.xml' in file:
        metas.append(file)

  images.sort()
  metas.sort()

  results = xml.etree.ElementTree.parse(FLAGS.result_xml).getroot()
  predictions = results.findall('image')

  print(str(images))
  print(str(predictions))

  total_iou = 0.0
  index = 0
  print('|----------------------------------------|')
  for image in images:
    print('Reading image ' + str(image))

    im = cv2.imread(FLAGS.image_dir + '/' + image, cv2.IMREAD_COLOR)
    meta = xml.etree.ElementTree.parse(FLAGS.image_dir + '/' + metas[index]).getroot()
    true_bndbox = {}
    true_bndbox['xmin'] = 0
    true_bndbox['xmax'] = 0
    true_bndbox['ymin'] = 0
    true_bndbox['ymax'] = 0
    if meta is not None:
      obj = meta.find('object')
      if obj is not None:
        box = obj.find('bndbox')
        if box is not None:
          true_bndbox['xmin'] = int(box.find('xmin').text)
          true_bndbox['xmax'] = int(box.find('xmax').text)
          true_bndbox['ymin'] = int(box.find('ymin').text)
          true_bndbox['ymax'] = int(box.find('ymax').text)


    bndbox = {}
    for meta in predictions:
      if meta.find('filename').text == image:
        box = meta.find('object').find('bndbox')
        bndbox['xmin'] = int(box.find('xmin').text)
        bndbox['ymin'] = int(box.find('ymin').text)
        bndbox['xmax'] = int(box.find('xmax').text)
        bndbox['ymax'] = int(box.find('ymax').text)

    cv2.rectangle(im, (bndbox['xmin'], bndbox['ymin']), (bndbox['xmax'],bndbox['ymax']), (0,255,0), 2)
    cv2.rectangle(im, (true_bndbox['xmin'], true_bndbox['ymin']), (true_bndbox['xmax'],true_bndbox['ymax']), (0,0,255), 2)

    cv2.imshow('image', im)
    cv2.waitKey(100)

    total_iou += bb_intersection_over_union(true_bndbox, bndbox)

    index += 1

  print("total_iou: " + str(total_iou/index))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--image_dir',
    type=str,
    default='',
    help='Absolute path to image directory.'
  )

  parser.add_argument(
    '--result_xml',
    type=str,
    default='',
    help='Absolute path to image directory.'
  )

  FLAGS, unparsed = parser.parse_known_args()

  display()