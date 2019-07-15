from math import log
import cv2
import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
import sys
import xml.etree.ElementTree

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.tfutils.varreplace import freeze_variables
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu

from dorefa import get_dorefa
from evaluate import *

DEMO_DATASET = 1

if DEMO_DATASET == 0:
	IMAGE_WIDTH = 640.0
	IMAGE_HEIGHT = 360.0
	BATCH_SIZE = 128
else:
	IMAGE_WIDTH = 640.0
	IMAGE_HEIGHT = 360.0
	BATCH_SIZE = 8

BITW = 1
BITA = 5
BITG = 32

MONITOR = 0
REAL_IMAGE = 1
DEBUG_DATA_PIPELINE = 0

NUM_SQUEEZE_FILTERS = 32
NUM_EXPAND_FILTERS = 96
PREDICT_CENTER = 0

resize_width = 320
resize_height = 176
grid_size = 16
num_grid_cells_x = int(resize_width/grid_size)
num_grid_cells_y = int(resize_height/grid_size)

classes = [
	['boat',0],
	['building',1],
	['car',2],
	['drone',3],
	['group',4],
	['horseride',5],
	['paraglider',6],
	['person',7],
	['riding',8],
	['truck',9],
	['wakeboard',10],
	['whale',11]
]

def line_intersection_union(line1_min, line1_max, line2_min, line2_max):
	intersection = 0
	intersect_state = 0
	if line1_min <= line2_min and line2_min < line1_max:
		if line2_max > line1_max:
			intersection = line1_max - line2_min
			intersect_state = 1
		else:
			intersection = line2_max - line2_min
			intersect_state = 2
	elif line2_min <= line1_min and line1_min < line2_max:
		if line1_max > line2_max:
			intersection = line2_max - line1_min
			intersect_state = 3
		else:
			intersection = line1_max - line1_min
			intersect_state = 4

	union = 0
	if intersection > 0:
		if intersect_state == 1:
			union = line2_max - line1_min
		elif intersect_state == 2:
			union = line1_max - line1_min
		elif intersect_state == 3:
			union = line1_max - line2_min
		else:
			union = line2_max - line2_min

	return intersection, union

def intersection(rect1, rect2):
	
	x_intersection, x_union = line_intersection_union(rect1['xmin'], rect1['xmax'], rect2['xmin'], rect2['xmax'])
	y_intersection, y_union = line_intersection_union(rect1['ymin'], rect1['ymax'], rect2['ymin'], rect2['ymax'])

	intersection = x_intersection*y_intersection
	union = x_union*y_union

	if intersection > 0:
		scaled = float(intersection)/(grid_size*grid_size)
		if scaled > 0.1:
			return scaled
		else:
			return 0
	else:
		return 0

class DAC_Dataset(RNGDataFlow):
	def __init__(self, dataset_dir, train, all_classes):
		self.images = []
		
		if all_classes == 1:
			for directory in listdir(dataset_dir):
				for file in listdir(dataset_dir + '/' + directory):
					if '.jpg' in file:
						for c in classes:
							if c[0] in directory:
								label = c[1]
								break
						self.images.append([dataset_dir + '/' + directory + '/' + file, label])
		else:
			for file in listdir(dataset_dir):
				if '.jpg' in file:
					self.images.append([dataset_dir + '/' + file, 0])

		shuffle(self.images)
		if train == 0:
			self.images = self.images[0:1000]

	def get_data(self):
		for image in self.images:
			xml_name = image[0].replace('jpg','xml')

			im = cv2.imread(image[0], cv2.IMREAD_COLOR)
			im = cv2.resize(im, (resize_width, resize_height))
			im = im.reshape((resize_height, resize_width, 3))

			meta = None
			if os.path.isfile(image[0].replace('jpg','xml')):
				meta = xml.etree.ElementTree.parse(xml_name).getroot()

			label = np.array(image[1])

			bndbox = {}
			bndbox['xmin'] = 0
			bndbox['xmax'] = 0
			bndbox['ymin'] = 0
			bndbox['ymax'] = 0

			if meta is not None:
				obj = meta.find('object')
				if obj is not None:
					box = obj.find('bndbox')
					if box is not None:
						bndbox['xmin'] = int(box.find('xmin').text.split('.')[0])
						bndbox['xmax'] = int(box.find('xmax').text.split('.')[0])
						bndbox['ymin'] = int(box.find('ymin').text.split('.')[0])
						bndbox['ymax'] = int(box.find('ymax').text.split('.')[0])

						bndbox['xmin'] = int(bndbox['xmin']*(resize_width/IMAGE_WIDTH))
						bndbox['xmax'] = int(bndbox['xmax']*(resize_width/IMAGE_WIDTH))
						bndbox['ymin'] = int(bndbox['ymin']*(resize_height/IMAGE_HEIGHT))
						bndbox['ymax'] = int(bndbox['ymax']*(resize_height/IMAGE_HEIGHT))

			iou = np.zeros( (num_grid_cells_y, num_grid_cells_x) )
			iou_weights = np.zeros( (num_grid_cells_y, num_grid_cells_x) )
			max_iou = 0
			for h in range(0, num_grid_cells_y):
				for w in range(0, num_grid_cells_x):
					rect = {}
					rect['xmin'] = int(w*grid_size)
					rect['xmax'] = int((w+1)*grid_size)
					rect['ymin'] = int(h*grid_size)
					rect['ymax'] = int((h+1)*grid_size)

					iou[h,w] = intersection(rect, bndbox)
					if iou[h,w] > max_iou:
						max_iou = iou[h,w]

			if max_iou == 0:
				continue

			for h in range(0, num_grid_cells_y):
				for w in range(0, num_grid_cells_x):
					rect = {}
					rect['xmin'] = int(w*grid_size)
					rect['xmax'] = int((w+1)*grid_size)
					rect['ymin'] = int(h*grid_size)
					rect['ymax'] = int((h+1)*grid_size)

					if (iou[h,w] > 0.0):
						if (max_iou > 0):
							iou_weights[h,w] = iou[h,w]/max_iou
						else:
							print('max_iou is 0 at' + xml_name +'!!!')
							cv2.rectangle(im, (int(rect['xmin']),int(rect['ymin'])), (int(rect['xmax']),int(rect['ymax'])), (0,0,iou_weights[h,w]*255), 1)
							cv2.imshow('image', im)
							cv2.waitKey(1000)
						iou[h,w] = 1.0
					else:
						iou_weights[h,w] = 1.0;

					if DEBUG_DATA_PIPELINE == 1:
						if iou[h,w] > 0:
							cv2.rectangle(im, (int(rect['xmin']),int(rect['ymin'])), (int(rect['xmax']),int(rect['ymax'])), (0,0,iou_weights[h,w]*255), 1)

			iou = iou.reshape( (num_grid_cells_y, num_grid_cells_x, 1) )
			iou_weights = iou_weights.reshape( (num_grid_cells_y, num_grid_cells_x, 1) )

			valid = np.zeros((num_grid_cells_y, num_grid_cells_x, 4), dtype='float32')
			relative_bndboxes = np.zeros((num_grid_cells_y, num_grid_cells_x, 4), dtype='float32')
			for h in range(0, num_grid_cells_y):
				for w in range(0, num_grid_cells_x):
					if iou[h, w] > 0.0:
						valid[h,w,0] = 1.0
						valid[h,w,1] = 1.0
						valid[h,w,2] = 1.0
						valid[h,w,3] = 1.0

						if PREDICT_CENTER == 1:
							c_x = ((bndbox['xmax'] + bndbox['xmin'])/2.0)-w*grid_size
							c_y = ((bndbox['ymax'] + bndbox['ymin'])/2.0)-h*grid_size
							t_w = bndbox['xmax'] - bndbox['xmin']
							t_h = bndbox['ymax'] - bndbox['ymin']
							relative_bndboxes[h, w, 0] = c_x
							relative_bndboxes[h, w, 1] = c_y
							relative_bndboxes[h, w, 2] = t_w
							relative_bndboxes[h, w, 3] = t_h
							if DEBUG_DATA_PIPELINE == 1:
								c_x_scale = c_x + w*grid_size
								c_y_scale = c_y + h*grid_size
								print("c_x_scale: " + str(c_x_scale))
								print("c_y_scale: " + str(c_y_scale))
								cv2.rectangle(im, (int(c_x_scale),int(c_y_scale)), (int(c_x_scale+2),int(c_y_scale+2)), (0,255,0), 3)
						else:
							relative_bndboxes[h, w, 0] = bndbox['xmin'] - w*grid_size
							relative_bndboxes[h, w, 1] = bndbox['ymin'] - h*grid_size
							relative_bndboxes[h, w, 2] = bndbox['xmax'] - w*grid_size
							relative_bndboxes[h, w, 3] = bndbox['ymax'] - h*grid_size
					else:
						relative_bndboxes[h, w] = np.zeros(4)

			if DEBUG_DATA_PIPELINE == 1:
				cv2.rectangle(im, (bndbox['xmin'],bndbox['ymin']), (bndbox['xmax'],bndbox['ymax']), (255,0,0), 1)
				cv2.imshow('image', im)
				cv2.waitKey(2000)

			yield [im, label, iou, iou_weights, valid, relative_bndboxes]

	def size(self):
		return len(self.images)

class Model(ModelDesc):
	def _get_inputs(self):
		return [InputDesc(tf.float32, [None, resize_height, resize_width, 3], 'input'),
				InputDesc(tf.int32, [None], 'label'),
				InputDesc(tf.float32, [None, num_grid_cells_y, num_grid_cells_x, 1], 'ious'),
				InputDesc(tf.float32, [None, num_grid_cells_y, num_grid_cells_x, 1], 'ious_weights'),
				InputDesc(tf.float32, [None, num_grid_cells_y, num_grid_cells_x, 4], 'valids'),
				InputDesc(tf.float32, [None, num_grid_cells_y, num_grid_cells_x, 4], 'bndboxes')]

	def _build_graph(self, inputs):
		image, label, ious, ious_weights, valids, bndboxes = inputs
		image = tf.round(image)

		fw, fa, fg = get_dorefa(BITW, BITA, BITG)

		old_get_variable = tf.get_variable

		def monitor(x, name):
			if MONITOR == 1:
				return tf.Print(x, [x], message='\n\n' + name + ': ', summarize=1000, name=name)
			else:
				return x

		def new_get_variable(v):
			name = v.op.name
			# if not name.endswith('W') or 'conv1' in name or 'conv_obj' in name or 'conv_box' in name:
			if not name.endswith('W') or 'conv_obj' in name or 'conv_box' in name:
				return v
			else:
				logger.info("Quantizing weight {}".format(v.op.name))
				if MONITOR == 1:
					return tf.Print(fw(v), [fw(v)], message='\n\n' + v.name + ', Quantized weights are:', summarize=100)
				else:
					return fw(v)

		def activate(x):
			if BITA == 32:
				return tf.nn.relu(x)
			else:
				return fa(tf.nn.relu(x))

		def bn_activate(name, x):
			x = BatchNorm(name, x)
			x = monitor(x, name + '_noact_out')
			return activate(x)

		def halffire(name, x, num_squeeze_filters, num_expand_3x3_filters, skip):
			out_squeeze = Conv2D('squeeze_conv_' + name, x, out_channel=num_squeeze_filters, kernel_shape=1, stride=1, padding='SAME')
			out_squeeze = bn_activate('bn_squeeze_' + name, out_squeeze)
			out_expand_3x3 = Conv2D('expand_3x3_conv_' + name, out_squeeze, out_channel=num_expand_3x3_filters, kernel_shape=3, stride=1, padding='SAME')
			out_expand_3x3 = bn_activate('bn_expand_3x3_' + name, out_expand_3x3)
			if skip == 0:
				return out_expand_3x3
			else:
				return tf.add(x, out_expand_3x3)

		def halffire_noact(name, x, num_squeeze_filters, num_expand_3x3_filters):
			out_squeeze = Conv2D('squeeze_conv_' + name, x, out_channel=num_squeeze_filters, kernel_shape=1, stride=1, padding='SAME')
			out_squeeze = bn_activate('bn_squeeze_' + name, out_squeeze)
			out_expand_3x3 = Conv2D('expand_3x3_conv_' + name, out_squeeze, out_channel=num_expand_3x3_filters, kernel_shape=3, stride=1, padding='SAME')
			return out_expand_3x3

		def halffire_final(l, name):
			l = halffire('fire4' + name, l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = halffire('fire5' + name, l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = halffire('fire6' + name, l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = halffire('fire7' + name, l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			return l

		def decision(l, name):
			classify = Conv2D('conv_class' + name, l, out_channel=12, kernel_shape=1, stride=1, padding='SAME')
			classify = bn_activate('bn_class' + name, classify)
			classify = monitor(classify, 'conv_class_out' + name)
			logits = GlobalAvgPooling('pool_class' + name, classify)

			l = tf.concat([l, classify], axis=3)

			objdetect = Conv2D('conv_obj' + name, l, out_channel=1, kernel_shape=1, stride=1, padding='SAME')
			bndbox = Conv2D('conv_box' + name, l, out_channel=4, kernel_shape=1, stride=1, padding='SAME')

			return logits, objdetect, bndbox

		def first_layer(x):
			l = Conv2D('conv1', x, out_channel=16, kernel_shape=3, stride=1, padding='SAME')
			l = bn_activate('bn1', l)
			l = monitor(l, 'conv1_out')
			return l

		with 	remap_variables(new_get_variable), \
				argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity), \
				argscope(BatchNorm, decay=0.9, epsilon=1e-4):

			image = monitor(image, 'image_out')

			if DEMO_DATASET == 0:
				l = first_layer(image)
			else:
				l = tf.stop_gradient(first_layer(image))

			l = MaxPooling('pool1', l, shape=3, stride=2, padding='SAME')
			l = monitor(l, 'pool1_out')

			l = halffire('fire1', l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = monitor(l, 'fire1_out')

			l = MaxPooling('pool2', l, shape=3, stride=2, padding='SAME')
			l = monitor(l, 'pool2_out')

			l = halffire('fire2', l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = monitor(l, 'fire2_out')

			l = MaxPooling('pool3', l, shape=3, stride=2, padding='SAME')
			l = monitor(l, 'pool3_out')

			l = halffire('fire3', l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = monitor(l, 'fire3_out')

			l = MaxPooling('pool4', l, shape=3, stride=2, padding='SAME')
			l = monitor(l, 'pool4_out')

			l1 = halffire_final(l, '1')
			l1 = monitor(l1, 'final1_out')
			l2 = halffire_final(l, '2')
			l2 = monitor(l2, 'final2_out')
			l3 = halffire_final(l, '3')
			l3 = monitor(l3, 'final3_out')
			l4 = halffire_final(l, '4')
			l4 = monitor(l4, 'final4_out')

			logits1, objdetect1, bndbox1 = decision(l1, '1')
			logits2, objdetect2, bndbox2 = decision(l2, '2')
			logits3, objdetect3, bndbox3 = decision(l3, '3')
			logits4, objdetect4, bndbox4 = decision(l4, '4')

			# Classification
			logits = (logits1 + logits2 + logits3 + logits4)/4
			class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label), name='cross_entropy_loss')

			wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
			add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

			# Object Detection
			objdetect = (objdetect1 + objdetect2 + objdetect3 + objdetect4)/4
			objdetect = tf.identity(objdetect, name='objdetect_out')
			objdetect_loss = tf.losses.hinge_loss(labels=ious, logits=objdetect, weights=ious_weights)

			bndbox = (bndbox1 + bndbox2 + bndbox3 + bndbox4)/4
			bndbox = tf.identity(bndbox, name='bndbox_out')
			bndbox_loss = tf.losses.mean_squared_error(labels=bndboxes, predictions=tf.multiply(bndbox, valids, name='mult0'))

			if DEMO_DATASET == 0:
				cost = class_loss + 5*objdetect_loss + bndbox_loss
			else:
				cost = 1000*objdetect_loss + bndbox_loss

			add_moving_summary(
				class_loss,
				objdetect_loss,
				bndbox_loss,
				cost)

		self.cost = cost

		tf.get_variable = old_get_variable

	def _get_optimizer(self):
		# if DEMO_DATASET == 0:
		lr = tf.get_variable('learning_rate', initializer=1e-2, trainable=False)
		opt = tf.train.AdamOptimizer(lr, epsilon=1e-5)
		# else:
			# lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
			# opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
		return opt

def get_data(dataset_dir, train):
	if DEMO_DATASET == 0:
		all_classes = 1
	else:
		all_classes = 0
	ds = DAC_Dataset(dataset_dir, train, all_classes)
	ds = BatchData(ds, BATCH_SIZE, remainder=True)
	ds = PrefetchDataZMQ(ds, nr_proc=8, hwm=6)
	return ds

def get_config():
	logger.auto_set_dir()
	data_train = get_data(args.data, 1)
	data_test = get_data(args.data, 0)

	if DEMO_DATASET == 0:
		return TrainConfig(
			dataflow=data_train,
			callbacks=[
				ModelSaver(max_to_keep=0),
				HumanHyperParamSetter('learning_rate'),
				ScheduledHyperParamSetter('learning_rate', [(20, 1e-3), (40, 1e-4), (60, 1e-5), (80, 1e-6)])
				,InferenceRunner(data_test,
								[ClassificationError('wrong-top1', 'val-error-top1')])
				],
			model=Model(),
			max_epoch=100
		)
	else:
		return TrainConfig(
			dataflow=data_train,
			callbacks=[
				ModelSaver(max_to_keep=0),
				HumanHyperParamSetter('learning_rate'),
				ScheduledHyperParamSetter('learning_rate', [(100, 1e-3), (200, 1e-4), (300, 1e-5)])
				],
			model=Model(),
			max_epoch=400
		)

def run_image(model, sess_init, image_dir):
	print('Running image!')

	output_names = ['objdetect_out', 'bndbox_out']

	pred_config = PredictConfig(
		model=model,
		session_init=sess_init,
		input_names=['input'],
		output_names=output_names
	)
	predictor = OfflinePredictor(pred_config)
	
	images = []
	metas = []
	for file in listdir(image_dir):
		if '.jpg' in file:
			images.append(file)
		if '.xml' in file:
			metas.append(file)

	images.sort()
	metas.sort()

	THRESHOLD = 0
	index = 0
	for image in images:
		meta = xml.etree.ElementTree.parse(image_dir + '/' + metas[index]).getroot()
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
					true_bndbox['xmin'] = int(box.find('xmin').text.split('.')[0])
					true_bndbox['xmax'] = int(box.find('xmax').text.split('.')[0])
					true_bndbox['ymin'] = int(box.find('ymin').text.split('.')[0])
					true_bndbox['ymax'] = int(box.find('ymax').text.split('.')[0])
		index += 1

		im = cv2.imread(image_dir + '/' + image, cv2.IMREAD_COLOR)
		im = cv2.resize(im, (resize_width, resize_height))
		im = im.reshape((1, resize_height, resize_width, 3))

		outputs = predictor([im])

		im = cv2.imread(image_dir + '/' + image, cv2.IMREAD_COLOR)

		objdetect = outputs[0]
		bndboxes = outputs[1]

		max_pred = -100
		max_h = -1
		max_w = -1
		for h in range(0, objdetect.shape[1]):
			for w in range(0, objdetect.shape[2]):
				if objdetect[0, h, w] > max_pred:
					max_pred = objdetect[0, h, w]
					max_h = h
					max_w = w

		sum_labels= 0;
		bndbox = {}
		bndbox['xmin'] = 0
		bndbox['ymin'] = 0
		bndbox['xmax'] = 0
		bndbox['ymax'] = 0
		for h in range(0, objdetect.shape[1]):
			for w in range(0, objdetect.shape[2]):
				if (objdetect[0, h, w] > THRESHOLD and (h == max_h-1 or h == max_h or h == max_h+1) and (w == max_w-1 or w == max_w or w == max_w+1)) or (h == max_h and w == max_w):
					sum_labels += 1

					bndbox['xmin'] += int( (bndboxes[0,h,w,0] + w*grid_size) )
					bndbox['ymin'] += int( (bndboxes[0,h,w,1] + h*grid_size) )
					bndbox['xmax'] += int( (bndboxes[0,h,w,2] + w*grid_size) )
					bndbox['ymax'] += int( (bndboxes[0,h,w,3] + h*grid_size) )

					temp_xmin = int(  (bndboxes[0,h,w,0] + w*grid_size) *(IMAGE_WIDTH/resize_width) )
					temp_ymin = int(  (bndboxes[0,h,w,1] + h*grid_size) *(IMAGE_HEIGHT/resize_height) )
					temp_xmax = int(  (bndboxes[0,h,w,2] + w*grid_size) *(IMAGE_WIDTH/resize_width) )
					temp_ymax = int(  (bndboxes[0,h,w,3] + h*grid_size) *(IMAGE_HEIGHT/resize_height) )
					# cv2.rectangle(im, (temp_xmin,temp_ymin), (temp_xmax,temp_ymax), (255,0,0), 1)

		bndbox['xmin'] = int(bndbox['xmin']*(1/sum_labels))
		bndbox['ymin'] = int(bndbox['ymin']*(1/sum_labels))
		bndbox['xmax'] = int(bndbox['xmax']*(1/sum_labels))
		bndbox['ymax'] = int(bndbox['ymax']*(1/sum_labels))
		bndbox['xmin'] = int(bndbox['xmin']*(IMAGE_WIDTH/resize_width))
		bndbox['ymin'] = int(bndbox['ymin']*(IMAGE_HEIGHT/resize_height))
		bndbox['xmax'] = int(bndbox['xmax']*(IMAGE_WIDTH/resize_width))
		bndbox['ymax'] = int(bndbox['ymax']*(IMAGE_HEIGHT/resize_height))
		print("bndboxes: " + str(bndboxes[0,max_h,max_w]))

		bndbox2 = {}
		if PREDICT_CENTER == 1:
			c_x = bndboxes[0,max_h,max_w,0] + max_w*grid_size
			c_y = bndboxes[0,max_h,max_w,1] + max_h*grid_size
			t_w = bndboxes[0,max_h,max_w,2]
			t_h = bndboxes[0,max_h,max_w,3]
			c_x_scale = int(c_x*(IMAGE_WIDTH/resize_width))
			c_y_scale = int(c_y*(IMAGE_HEIGHT/resize_height))
			cv2.rectangle(im, (c_x_scale,c_y_scale), (c_x_scale+2,c_y_scale+2), (255,0,0), 3)
			bndbox2['xmin'] = int(c_x - t_w/2.0)
			bndbox2['xmax'] = int(c_x + t_w/2.0)
			bndbox2['ymin'] = int(c_y - t_h/2.0)
			bndbox2['ymax'] = int(c_y + t_h/2.0)
		else:
			bndbox2['xmin'] = int( bndboxes[0,max_h,max_w,0] + max_w*grid_size)
			bndbox2['ymin'] = int( bndboxes[0,max_h,max_w,1] + max_h*grid_size)
			bndbox2['xmax'] = int( bndboxes[0,max_h,max_w,2] + max_w*grid_size)
			bndbox2['ymax'] = int( bndboxes[0,max_h,max_w,3] + max_h*grid_size)

		bndbox2['xmin'] = int(bndbox2['xmin']*(IMAGE_WIDTH/resize_width))
		bndbox2['ymin'] = int(bndbox2['ymin']*(IMAGE_HEIGHT/resize_height))
		bndbox2['xmax'] = int(bndbox2['xmax']*(IMAGE_WIDTH/resize_width))
		bndbox2['ymax'] = int(bndbox2['ymax']*(IMAGE_HEIGHT/resize_height))

		print('----------------------------------------')
		print(str(max_h*(IMAGE_WIDTH/grid_size)+max_w))
		print('xmin: ' + str(bndbox2['xmin']))
		print('xmax: ' + str(bndbox2['xmax']))
		print('ymin: ' + str(bndbox2['ymin']))
		print('ymax: ' + str(bndbox2['ymax']))

		cv2.rectangle(im, (int(max_w*grid_size*(IMAGE_WIDTH/resize_width)),int(max_h*grid_size*(IMAGE_HEIGHT/resize_height))), (int((max_w+1)*grid_size*(IMAGE_WIDTH/resize_width)),int((max_h+1)*grid_size*(IMAGE_HEIGHT/resize_height))), (0,0,255), 1)
		cv2.rectangle(im, (true_bndbox['xmin'], true_bndbox['ymin']), (true_bndbox['xmax'], true_bndbox['ymax']), (255,0,0), 2)
		cv2.rectangle(im, (bndbox2['xmin'], bndbox2['ymin']), (bndbox2['xmax'],bndbox2['ymax']), (0,255,0), 2)

		cv2.imshow('image', im)
		cv2.imwrite('images_log/' + image, im)
		cv2.waitKey(100)

def run_single_image(model, sess_init, image):
	print('Running single image!')

	if MONITOR == 1:
		monitor_names = ['image_out', 'conv1_out', 'pool1_out', 'fire1_out', 'pool2_out', 'pool3_out', 'final4_out']
	else:
		monitor_names = []
	output_names = ['objdetect_out', 'bndbox_out']
	output_names.extend(monitor_names)

	pred_config = PredictConfig(
		model=model,
		session_init=sess_init,
		input_names=['input'],
		output_names=output_names
	)
	predictor = OfflinePredictor(pred_config)

	if REAL_IMAGE == 1:
		im = cv2.imread(image, cv2.IMREAD_COLOR)
		im = cv2.resize(im, (resize_width, resize_height))
		cv2.imwrite('test_image.png', im)
		im = im.reshape((1, resize_height, resize_width, 3))
	else:
		im = np.zeros((1, resize_height, resize_width, 3))
		k = 0
		for h in range(0, resize_height):
			for w in range(0,resize_width):
				for c in range (0,3):
					# im[0][h][w][c] = 0
					im[0][h][w][c] = k%256
					k += 1

	outputs = predictor([im])

	objdetect = outputs[0]
	bndboxes = outputs[1]

	max_pred = -100
	max_h = -1
	max_w = -1
	for h in range(0, objdetect.shape[1]):
		for w in range(0, objdetect.shape[2]):
			print(str(objdetect[0, h, w]))
			if objdetect[0, h, w] > max_pred:
				max_pred = objdetect[0, h, w]
				max_h = h
				max_w = w
	bndbox2 = {}
	if PREDICT_CENTER == 1:
		c_x = bndboxes[0,max_h,max_w,0] + max_w*grid_size
		c_y = bndboxes[0,max_h,max_w,1] + max_h*grid_size
		t_w = bndboxes[0,max_h,max_w,2]
		t_h = bndboxes[0,max_h,max_w,3]
		c_x_scale = int(c_x*(IMAGE_WIDTH/resize_width))
		c_y_scale = int(c_y*(IMAGE_HEIGHT/resize_height))
		cv2.rectangle(im, (c_x_scale,c_y_scale), (c_x_scale+2,c_y_scale+2), (255,0,0), 3)
		bndbox2['xmin'] = int(c_x - t_w/2.0)
		bndbox2['xmax'] = int(c_x + t_w/2.0)
		bndbox2['ymin'] = int(c_y - t_h/2.0)
		bndbox2['ymax'] = int(c_y + t_h/2.0)
	else:
		bndbox2['xmin'] = int( bndboxes[0,max_h,max_w,0] + max_w*grid_size)
		bndbox2['ymin'] = int( bndboxes[0,max_h,max_w,1] + max_h*grid_size)
		bndbox2['xmax'] = int( bndboxes[0,max_h,max_w,2] + max_w*grid_size)
		bndbox2['ymax'] = int( bndboxes[0,max_h,max_w,3] + max_h*grid_size)
	bndbox2['xmin'] = int(bndbox2['xmin']*(IMAGE_WIDTH/resize_width))
	bndbox2['ymin'] = int(bndbox2['ymin']*(IMAGE_HEIGHT/resize_height))
	bndbox2['xmax'] = int(bndbox2['xmax']*(IMAGE_WIDTH/resize_width))
	bndbox2['ymax'] = int(bndbox2['ymax']*(IMAGE_HEIGHT/resize_height))

	# if REAL_IMAGE == 1:
	# 	im = cv2.imread(image, cv2.IMREAD_COLOR)
	# 	cv2.rectangle(im, (bndbox2['xmin'], bndbox2['ymin']), (bndbox2['xmax'],bndbox2['ymax']), (0,255,0), 2)
	# 	cv2.imshow('image', im)
	# 	cv2.waitKey(2000)

	print('max_h: ' + str(max_h))
	print('max_w: ' + str(max_w))
	print('objdetect.shape: ' + str(objdetect.shape))
	print('bndboxes: ' + str(bndboxes[0,max_h,max_w]))
	print('bndbox2: ' + str([bndbox2['xmin'], bndbox2['ymin'], bndbox2['xmax'],bndbox2['ymax']]))
	
	index = 2
	for o in monitor_names:
		print(o + ', shape: ' + str(outputs[index].shape) )

		if 'image' not in o:
			# outputs[index] = np.multiply(outputs[index], 2**SCALE_BITS).astype('int32')
			print(str(outputs[index]))

		if len(outputs[index].shape) == 4:
			file_name = o.split('/')[-1]
			print('Writing file... ' + file_name)
			if not os.path.exists('./log'):
				os.makedirs('./log')
			with open('./log/' + file_name + '.log', 'w') as f:
				for sample in range(0, outputs[index].shape[0]):
					for h in range(0, outputs[index].shape[1]):
						for w in range(0, outputs[index].shape[2]):
							res = ''
							for c in range(0, outputs[index].shape[3]):
								if 'image' in file_name:
									res = hexFromInt( int(outputs[index][sample, h, w, c]), 8 ) + '_' + res
								elif 'noact' in file_name:
									temp = (2**FACTOR_SCALE_BITS)*outputs[index][sample, h, w, c]
									res = hexFromInt( int(temp), 32 ) + '_' + res
								else:
									res = hexFromInt( int(outputs[index][sample, h, w, c]), BITA) + '_' + res
							f.write('0x' + res + '\n')
		index += 1

def run_batch(model, sess_init, image_dir):
	print('Running batch!')

	output_names = ['objdetect_out', 'bndbox_out']

	pred_config = PredictConfig(
		model=model,
		session_init=sess_init,
		input_names=['input'],
		output_names=output_names
	)
	predictor = OfflinePredictor(pred_config)

	folders = []
	folders = [x[0] for x in os.walk(image_dir)]
	folders.sort()

	batch_iou = 0
	batch_cnt = 0

	for folder in folders:
		images = []
		metas = []

		for file in listdir(folder):
			if '.jpg' in file:
				images.append(file)
			if '.xml' in file:
				metas.append(file)

		images.sort()
		metas.sort()

		THRESHOLD = 0
		index = 0
		folder_iou = 0
		for image in images:
			meta = xml.etree.ElementTree.parse(folder + '/' + metas[index]).getroot()
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
			index += 1

			im = cv2.imread(folder + '/' + image, cv2.IMREAD_COLOR)
			im = cv2.resize(im, (resize_width, resize_height))
			im = im.reshape((1, resize_height, resize_width, 3))

			outputs = predictor([im])


			objdetect = outputs[0]
			bndboxes = outputs[1]

			max_pred = -100
			max_h = -1
			max_w = -1
			for h in range(0, objdetect.shape[1]):
				for w in range(0, objdetect.shape[2]):
					if objdetect[0, h, w] > max_pred:
						max_pred = objdetect[0, h, w]
						max_h = h
						max_w = w

			bndbox = {}
			if PREDICT_CENTER == 1:
				c_x = bndboxes[0,max_h,max_w,0] + max_w*grid_size
				c_y = bndboxes[0,max_h,max_w,1] + max_h*grid_size
				t_w = bndboxes[0,max_h,max_w,2]
				t_h = bndboxes[0,max_h,max_w,3]
				c_x_scale = int(c_x*(IMAGE_WIDTH/resize_width))
				c_y_scale = int(c_y*(IMAGE_HEIGHT/resize_height))
				cv2.rectangle(im, (c_x_scale,c_y_scale), (c_x_scale+2,c_y_scale+2), (255,0,0), 3)
				bndbox['xmin'] = int(c_x - t_w/2.0)
				bndbox['xmax'] = int(c_x + t_w/2.0)
				bndbox['ymin'] = int(c_y - t_h/2.0)
				bndbox['ymax'] = int(c_y + t_h/2.0)
			else:
				bndbox['xmin'] = int( bndboxes[0,max_h,max_w,0] + max_w*grid_size)
				bndbox['ymin'] = int( bndboxes[0,max_h,max_w,1] + max_h*grid_size)
				bndbox['xmax'] = int( bndboxes[0,max_h,max_w,2] + max_w*grid_size)
				bndbox['ymax'] = int( bndboxes[0,max_h,max_w,3] + max_h*grid_size)

			bndbox['xmin'] = int(bndbox['xmin']*(IMAGE_WIDTH/resize_width))
			bndbox['ymin'] = int(bndbox['ymin']*(IMAGE_HEIGHT/resize_height))
			bndbox['xmax'] = int(bndbox['xmax']*(IMAGE_WIDTH/resize_width))
			bndbox['ymax'] = int(bndbox['ymax']*(IMAGE_HEIGHT/resize_height))

			# intersection over union
			iou = bb_intersection_over_union(true_bndbox,bndbox)
			folder_iou += iou
			batch_iou += iou
			batch_cnt += 1

			if MONITOR == 1:
				print('----------------------------------------')
				print(str(max_h*14+max_w))
				print('bndbox = [' + str(bndbox['xmin']) + ',' + str(bndbox['xmax']) + ',' + str(bndbox['ymin']) + ',' + str(bndbox['ymax']) + ']')
				print('IoU: ' + str(iou))

			# im = cv2.imread(folder + '/' + image, cv2.IMREAD_COLOR)
			# cv2.rectangle(im, (int(max_w*grid_size*(IMAGE_WIDTH/resize_width)),int(max_h*grid_size*(IMAGE_HEIGHT/resize_height))), (int((max_w+1)*grid_size*(IMAGE_WIDTH/resize_width)),int((max_h+1)*grid_size*(IMAGE_HEIGHT/resize_height))), (0,0,255), 1)
			# cv2.rectangle(im, (true_bndbox['xmin'], true_bndbox['ymin']), (true_bndbox['xmax'], true_bndbox['ymax']), (255,0,0), 2)
			# cv2.rectangle(im, (bndbox['xmin'], bndbox['ymin']), (bndbox['xmax'],bndbox['ymax']), (0,255,0), 2)
			# cv2.imshow('image', im)
			# cv2.imwrite('images_log/' + image, im)
			# cv2.waitKey(800)

		if index != 0:
			print('##### ' + folder)
			print('IoU = ' + str(folder_iou / index))
			print('cnt = ' + str(index))
	if batch_cnt != 0:
		print('------------------------')
		print('Global IoU = ' + str(batch_iou / batch_cnt))
		print('Global cnt = ' + str(batch_cnt))

def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA['xmin'], boxB['xmin'])
	yA = max(boxA['ymin'], boxB['ymin'])
	xB = min(boxA['xmax'], boxB['xmax'])
	yB = min(boxA['ymax'], boxB['ymax'])
 
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	boxAArea = (boxA['xmax'] - boxA['xmin'] + 1) * (boxA['ymax'] - boxA['ymin'] + 1)
	boxBArea = (boxB['xmax'] - boxB['xmin'] + 1) * (boxB['ymax'] - boxB['ymin'] + 1)
 
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

	return iou

def dump_weights(meta, model, output):
	fw, fa, fg = get_dorefa(BITW, BITA, BITG)

	with tf.Graph().as_default() as G:
		tf.train.import_meta_graph(meta)

		init = get_model_loader(model)
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		sess.run(tf.global_variables_initializer())
		init.init(sess)

		with sess.as_default():
			if output:
				if output.endswith('npy') or output.endswith('npz'):
					varmanip.dump_session_params(output)
				else:
					var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
					var.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
					var_dict = {}
					for v in var:
						name = varmanip.get_savename_from_varname(v.name)
						var_dict[name] = v
					logger.info("Variables to dump:")
					logger.info(", ".join(var_dict.keys()))
					saver = tf.train.Saver(
						var_list=var_dict,
						write_version=tf.train.SaverDef.V2)
					saver.save(sess, output, write_meta_graph=False)

			network_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			network_model.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))

			target_frequency = 200000000
			target_FMpS = 300
			# non_quantized_layers = ['conv1/Conv2D', 'conv_obj/Conv2D', 'conv_box/Conv2D']
			non_quantized_layers = [
			'conv_obj1/Conv2D',
			'conv_box1/Conv2D',
			'conv_obj2/Conv2D',
			'conv_box2/Conv2D',
			'conv_obj3/Conv2D',
			'conv_box3/Conv2D',
			'conv_obj4/Conv2D',
			'conv_box4/Conv2D']

			json_out, layers_list, max_cycles = generateLayers(sess, BITA, BITW, non_quantized_layers, target_frequency, target_FMpS)
			
			achieved_FMpS = target_frequency/max_cycles

			if DEMO_DATASET == 0:
				generateConfig(layers_list, 'halfsqueezenet-config.h')
				genereateHLSparams(layers_list, network_model, 'halfsqueezenet-params.h', fw)
			else:
				generateConfig(layers_list, 'halfsqueezenet-config_demo.h')
				genereateHLSparams(layers_list, network_model, 'halfsqueezenet-params_demo.h', fw)

			print('|---------------------------------------------------------|')
			print('target_FMpS: ' + str(target_FMpS) )
			print('achieved_FMpS: ' + str(achieved_FMpS) )

if __name__ == '__main__':
	print('Start')

	parser = argparse.ArgumentParser()
	parser.add_argument('dump2_train1_test0', help='dump(2), train(1) or test(0)')
	parser.add_argument('--model', help='model file')
	parser.add_argument('--meta', help='metagraph file')
	parser.add_argument('--output', help='output for dumping')
	parser.add_argument('--gpu', help='the physical ids of GPUs to use')
	parser.add_argument('--data', help='DAC dataset dir')
	parser.add_argument('--run', help='directory of images to test')
	parser.add_argument('--weights', help='weights file')
	args = parser.parse_args()

	print('Using GPU ' + str(args.gpu))

	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	print(str(args.dump2_train1_test0))

	if args.dump2_train1_test0 == '1':
		if args.data == None:
			print('Provide DAC dataset path with --data')
			sys.exit()

		config = get_config()
		if args.model:
			config.session_init = SaverRestore(args.model)

		SimpleTrainer(config).train()

	elif args.dump2_train1_test0 == '0':
		if args.run == None:
			print('Provide images with --run ')
			sys.exit()
		if args.weights == None:
			print('Provide weights file (.npy) for testing!')
			sys.exit()

		assert args.weights.endswith('.npy')
		run_image(Model(), DictRestore(np.load(args.weights, encoding='latin1').item()), args.run)

	elif args.dump2_train1_test0 == '2':
		if args.meta == None:
			print('Provide meta file (.meta) for dumping')
			sys.exit()
		if args.model == None:
			print('Provide model file (.data-00000-of-00001) for dumping')
			sys.exit()

		dump_weights(args.meta, args.model, args.output)

	elif args.dump2_train1_test0 == '3':
		if args.run == None:
			print('Provide image with --run ')
			sys.exit()
		if args.weights == None:
			print('Provide weights file (.npy) for testing!')
			sys.exit()

		assert args.weights.endswith('.npy')
		run_single_image(Model(), DictRestore(np.load(args.weights, encoding='latin1').item()), args.run)

	elif args.dump2_train1_test0 == '4':
		if args.run == None:
			print('Provide images with --run ')
			sys.exit()
		if args.weights == None:
			print('Provide weights file (.npy) for testing!')
			sys.exit()

		assert args.weights.endswith('.npy')
		run_batch(Model(), DictRestore(np.load(args.weights, encoding='latin1').item()), args.run)