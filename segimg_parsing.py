#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
For Parsing
"""

import os
import sys

from PyQt4 import QtGui
from PyQt4.QtGui import *
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import time
from collections import OrderedDict
import operator
from PIL import ImageFont
import cv2

from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage import measure
from my_pic import MyPic

import scipy.io as sio

import functools
import logging

def draw_rect(img, b, color=None):
	draw = ImageDraw.Draw(img, 'RGBA')
	if color is None:
		color = (255, 0, 0, 100)
	draw.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], fill=color)


def draw_text(img, s, pos, ft_size, color):
	draw = ImageDraw.Draw(img, 'RGBA')
	font = ImageFont.truetype("arial.ttf", ft_size)
	draw.text(pos, s, color, font)


# io part
#import xmltodict
import json


class MyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(MyEncoder, self).default(obj)

def write_json(save_fp, d):
	with open(save_fp, 'w') as outfile:
		json.dump(d, outfile, cls=MyEncoder, indent=4, sort_keys=True, separators=(',', ':'))


def read_json(fp):
	with open(fp) as f:
		d = json.load(f)
		return d
	return None


def read_xml(fp):
	with open(fp) as f:
		d = xmltodict.parse(f)
		return d
	return None


def read_image_arr(p):
	tmp = np.array(Image.open(p))
	return tmp


def debug_read_image_arr(p):
	tmp = np.array(Image.open(p).resize((800, 500)))
	return tmp


def cvtrgb2qtimage(arr):
	H, W = arr.shape[0], arr.shape[1]
	return QtGui.QImage(arr, W, H, W * 3, QtGui.QImage.Format_RGB888)


def getfilelist(searchdir, regexp='.*'):
	"""
	Find all the file that match regexp where regexp is the regular expression.
	"""
	import os
	import re
	if not os.path.exists(searchdir):
		return []
	allfile = os.listdir(searchdir)
	pattern = re.compile(regexp)
	filelist = []
	for f in allfile:
		tmp = pattern.match(f)
		if tmp is not None:
			filelist.append(f)
	return filelist


def ensure_dir(d):
	if not os.path.exists(d):
		os.makedirs(d)


class QPlainTextEditLogger(logging.StreamHandler):
	def __init__(self, parent):
		super(QPlainTextEditLogger, self).__init__()
		self.widget = QtGui.QPlainTextEdit(parent)
		self.widget.setReadOnly(True)

	def emit(self, record):
		msg = self.format(record)
		self.widget.insertPlainText(msg)
		self.widget.verticalScrollBar().setSliderPosition(
			self.widget.verticalScrollBar().maximum())

	def write(self, m):
		pass


class SegPic(MyPic):
	class BiBlock:
		def __init__(self):
			self.BI = None  # (0 or 255) image
			self.boundary = []

	class LaneLine:
		def __init__(self):
			self.line_type = None
			self.points = []
			self.break_idx = []

	class Collected:
		def __init__(self, W, H):
			self.center = [] 	# (y, x)s
			self.leftE = []		# (y, x)s; left edge (left Bound if edge doesn't exist)
			self.rightE = []	# (y, x)s; right edge (right Bound if ...)
			self.sub_bbox = [] 	# [[(TLx, TLy), (BRx, BRy)], [], []...]	 rects of each center points
			self.common_bbox = [[W-1, H-1], [0, 0]] 	#[[TLx, TLy], [BRx, BRy]  rect which contains all sub_bboxes

	def __init__(self, *arg):
		super(SegPic, self).__init__(*arg)
		self.skip = False  # True if we don't annotate this image temporarily
		self.collect_points = []  # points used in annotate super pixel
		self.collect_poly_points = []
		self.zoomed_collect_points = []
		self.setMouseTracking(True)
		self.seg_index = None  # superpixel index
		self.seg_arr = None  # The classification label
		self.seg_disp = None  # The color map for disp
		self.seg_disp_polygon = None
		self.ref_pic = None
		self.color_map = None
		self.label_names = None
		self.label_index = None
		self.alpha = 0.35
		self.current_label = 0
		self.seg_params = None
		self.ignore_label = 255

		### Zoom in/out ###
		self.zRate = 6
		self.Zoomed = False		# True if the image is zoomed (key_V pressed)
		self.visualP = []
		# zoom position; position of current canvas win in the large zoomed window
		self.zoom_pos = [] 	 # [left, upper, right, lower]
		self.zoomed_BI = None	# temp final_BI layer for zoomed image
		self.zoomed_collect = None # temp Collected() obj to store points' info on zoomed image
		self.ZoomedModified = True	# True if final_BI has been modified during zoom in
		self.i_mark = 0
		self.large_ref_img = None

		# key -- line_label; Value -- [] of 2; 
		# value[0] -- BIBlock_stk; value[1] -- final_BI (0 or label) binary image 

		self.grabcut = None

		# self.lineSeg_Disp = None
		self.lineSegDisp_dict = {}   # dict for all line_type's seg_disp arr (painted with its corresponding color_fill)
		self.ImgLoaded = False  # True if image has already been loaded from folder
		self.ori_image = None  	# orignal sized img (for zoom out)


	def set_reference_pic(self, ref_pic):
		self.ref_pic = ref_pic

		# seg_arr init to ignore_label
		if not (ref_pic.img_arr is None):
			self.seg_arr = np.zeros(ref_pic.img_arr.shape[:2], dtype=np.uint8)
			self.seg_arr[:] = self.ignore_label
			# init line_dict
			self.line_set = set()
		print ">>>>>>>>>> seg_arr.shape: ", self.seg_arr.shape


	def set_seg_params(self, seg_params):
		self.seg_params = seg_params

	def init_seg(self):
		ori_img = self.ref_pic.img_arr
		self.seg_arr = np.zeros(ori_img.shape[:2], dtype=np.uint8)
		self.seg_arr[:] = self.ignore_label
		# init line_dict
		self.line_set = set()


	def update_disp(self):
		"""
				1) update self.seg_disp (current round) based on seg_arr and color_map
				2) update self.img_arr on the basis of self.seg_disp (current) and ref_pic.img_arr (containing
				previously labeled info)
		"""
		print "Update_Disp..."
		# print '--self.line_set: ', self.line_set
		ref_pic = self.ref_pic
		ori_img = ref_pic.img_arr

		if not (ori_img is None):
			# 1) update seg area according to seg_disp
			if (self.parent().edit_method == "segmentation" and self.ImgLoaded == False):
				print "############## ImgLoaded == False"
				r, c = self.seg_arr.shape[0], self.seg_arr.shape[1]
				self.seg_disp = self.color_map[self.seg_arr.ravel(), :].reshape(
					(r, c, 3))
				self.img_arr = np.array(
					ref_pic.img_arr * (1 - self.alpha) + self.alpha * self.seg_disp, dtype=np.uint8)

			# 2) draw lines on img_arr
			# elif (self.parent().edit_method == "polygon"):
			else:
				# print "@@@@@@@@@@@@@ editmethod: ", self.parent().edit_method
				self.update_segDisp_from_segArr()
				# print "########## seg_disp_polygon.shape: ", self.seg_disp_polygon.shape
				# print "########## self.ref_pic.img_arr.shape: ", self.ref_pic.img_arr.shape
				if self.Zoomed == True:
					## zoom (resize and crop) lineSeg_Disp					
					img_segDispPolygon = Image.fromarray(self.seg_disp_polygon)
					largeImg_segDispPolygon = img_segDispPolygon.resize((self.w * self.zRate, self.h * self.zRate), Image.NEAREST)
					cropped_segDispPolygon = largeImg_segDispPolygon.crop(tuple(self.zoom_pos))
					seg_disp = np.array(cropped_segDispPolygon)
				else:
					seg_disp = self.seg_disp_polygon

				self.img_arr = np.array(
					ref_pic.img_arr * (1 - self.alpha) + self.alpha * seg_disp, dtype=np.uint8)


	def update_segDisp_from_segArr(self):
		print ">>>>>>>>> update_segDisp_from_segArr"
		self.seg_disp_polygon = self.ori_img.copy() ## always draw on original img

		seg_dispImg = Image.fromarray(self.seg_disp_polygon)
		draw = ImageDraw.Draw(seg_dispImg, mode='RGB')
		mask = np.zeros(self.seg_arr.shape, np.uint8)
		for label_idx in self.label_index:
			if label_idx == 255:
				continue
			mask[:] = 0;
			mask[self.seg_arr == label_idx] = 255
			label_color = tuple(self.color_map[label_idx])
			maskImg = Image.fromarray(mask)
			draw.bitmap((0, 0), maskImg, fill=tuple(label_color))

		# seg_dispImg.show()
		self.seg_disp_polygon = np.array(seg_dispImg)


	def _draw_dots(self, points, color):
		# draw_dots using qt canvas
		r, g, b, o = color		

		if self.img_arr is None:
			self.img_arr = self.ref_pic.img_arr
		qImg = cvtrgb2qtimage(self.img_arr)
		image = QtGui.QPixmap(qImg)

		radx = 2
		rady = 2
		draw = QtGui.QPainter()
		draw.begin(image)
		draw.setBrush(QtGui.QColor(r, g, b, o))
		for p in points:
			center = QtCore.QPoint(p[1], p[0])
			draw.drawEllipse(center, radx, rady)
		draw.end()
		self.setPixmap(image)   


	def set_seg_config(self, ref_pic, label_type, seg_method):
		"""
		This function is used for initializing the segmentation part
		according to the label_type and seg_method
		"""
		self.set_reference_pic(ref_pic)
		# self.set_ref_img(ref_pic)
		self.label_type = label_type
		self.load_label_info(label_type)
		# self.load_line_info()
		self.seg_method = seg_method
		self.init_seg()  # dont use label_type; init seg_arr with ignored label
		self.update_disp()  # dont use label_type
		self.update()  # dont use label_type
	# def update(self):
	#     super(SegPic, self).update()

	
	def _point_small2large(self, pointS_list):
		# transform points on small img --> points on large img and reset origin as small img's TL corner
		# point - [x, y]
		res = []
		for p in pointS_list:
			pointL = [int(p[0] * self.zRate) - self.zoom_pos[0], int(p[1] * self.zRate) - self.zoom_pos[1]]
			res.append(pointL)
		return res


	def _large_point_inbound(self, points, break_i):
		# get the first/last point out of bound, replace it with a point on boundary by forming a line
		# between the point and its nearest in-bound point
		# pointS -- (x, y)

		outofbound = []
		res = []
		break_res = []

		for p in points:
			outofbound.append(self._point_outofbound(p))

		j = 0
		for i in range(len(points)):
			if outofbound[i] == False:
				res.append(points[i])
				if i in break_i:
					break_res.append(len(res)-1)

			if outofbound[i] == True and (i+1 < len(points)) and outofbound[i+1] == False:
				p_boarder = self._get_point_on_boarder(points[i], points[i+1])
				res.append(p_boarder)
				if i in break_i:
					break_res.append(len(res)-1)

			if outofbound[i] == True and (i-1 >= 0) and outofbound[i-1] == False:
				p_boarder = self._get_point_on_boarder(points[i], points[i-1])
				res.append(p_boarder)

		return (res, break_res)


	def _point_large2small(self, pointL_list):
		# pointL -- [x, y]
		out = []
		for p in pointL_list:
			xS = int((p[0] + self.zoom_pos[0])/self.zRate)
			yS = int((p[1] + self.zoom_pos[1])/self.zRate)
			out.append( [xS, yS] )
		return out


	def _point_outofbound(self, p):
		# p -- [x, y]
		if p[0] < 0 or p[0] >= self.w:
			return True
		if p[1] < 0 or p[1] >= self.h:
			return True
		return False


	def _get_point_on_boarder(self, p_out, p_in):
		# p -- [x, y]
		## case1. line is vertical
		if float(p_out[0] - p_in[0]) == 0:
			if p_out[1] < 0:
				return [p_out[0], 0]
			else:
				return [p_out[0], self.h-1]

		## case2.
		k = float(p_out[1] - p_in[1])/float(p_out[0] - p_in[0])
		c = p_out[1] - k * p_out[0]
		## 2.1) intersect with y = 0
		x = int(-1 * c/k)
		if (x - p_out[0]) * (x - p_in[0]) <= 0:
			return [x, 0]
		## 2.2) intersect with y = self.h-1
		x = int((self.h - 1 - c)/k)
		if (x - p_out[0]) * (x - p_in[0]) <= 0:
			return [x, self.h-1]
		## 2.3) intersect with x = 0
		if (c - p_out[1]) * (c - p_in[1]) <= 0:
			return [0, c]
		## 2.3) intersect with x = self.w-1
		y = int(c + k * (self.w - 1))
		if (y - p_out[1]) * (y - p_in[1]) <= 0:
			return [self.w-1, y]


	@classmethod
	def read_label_name(cls, fp):
		with open(fp, 'r') as f:
			lines = f.readlines()
			return lines
		return None


	def load_label_info(self, label_type):
		# load color map
		# color_map (num_labels * 3); color_map[2] -- color for label_2
		mydir = os.path.dirname(os.path.realpath(__file__))
		# e.g. color_cityscape_exp_1
		self.color_map = sio.loadmat(
			'{}/data/color_{}.mat'.format(mydir, label_type))['colors']
		self.color_map[255, :] = 255   # ignore_label = 255
		fp = os.path.join(mydir, 'data', '{}_labels.txt'.format(
			label_type))  # e.g. cityscape_exp_1_labels.txt
		self.label_names = self.read_label_name(fp)
		self.label_names.append('ignore')

		self.label_index = range(len(self.label_names))
		self.label_index.append(255)


	def undo_seg(self):
		"""
		pop from biblock_stk; set the all 1s in bi_rect as 0s
		!!! This method also updated lineSeg_BI
		"""
		## 1) check if cur_line_label exists in final_BI
		##    we shouldn't allow undo if user selected line_type does not exists on img,
		##	  because he/she will not see the change directly.
		if self.cur_line_label not in self.final_BI:
			return

		if len(self.BiBlock_stk) > 0:
			biblock = self.BiBlock_stk.pop(-1)
			rect = biblock.boundary
			bi = biblock.BI
			# Image.fromarray(bi).show()
			tmp = np.zeros(self.final_BI.shape, np.uint8)
			tmp[rect[0] : rect[1], rect[2] : rect[3]] = bi
				
			self.final_BI[tmp != 0] = 0
			# keep self.final_ID sync with self.final_BI
			self.final_ID[tmp != 0] = -self.int8_to_uint8_OFFSET

			# self._update_ID2label(self.cur_line_ID, self.cur_line_label)
			self._ID2label_dict_del(self.cur_line_ID, self.cur_line_label)

			## update lineSegDisp_dict (must after lineSeg_BI_dict updated!!!)
			# self.lineSegDisp_dict[self.cur_line_type2] = self._get_lineSeg_Disp(self.cur_line_type2)
			self.update_disp()
			self.update()


	def event(self, event):
		if self.parent().mode != "edit":
			return QtGui.QLabel.event(self, event)

		et = event.type()
		edit_method = self.parent().edit_method
		# print self.parent()

		if self.Zoomed == True:
			collect_poly_points = self.zoomed_collect_points
		else:
			collect_poly_points = self.collect_poly_points

		if (et == QtCore.QEvent.MouseButtonPress):
			# RIGHT Click -- undo last point opt
			if (event.buttons() == QtCore.Qt.RightButton):
				if edit_method == 'segmentation':
					# right-mouse-button to reset the superpixel
					self.reset_current_superpixel(
						event.pos().x(), event.pos().y())

				elif edit_method == 'polygon':
					if collect_poly_points:
						last_point = collect_poly_points.pop()
						# print "POP POINT!"
						self.update_disp()

						k = len(collect_poly_points)
						# print "len of collect_pp = ", k
						if k > 1:
							self.draw_polyline(
								collect_poly_points, (255, 0, 0))
						self.update()


			# LEFT Click
			elif (event.buttons() == QtCore.Qt.LeftButton):
				if self.parent().edit_method == "polygon":
					# [(p1x, p1y), (p2x, p2y), ...]
					collect_poly_points.append(
						[event.pos().x(), event.pos().y()])

					if len(collect_poly_points) > 1:
						p1 = collect_poly_points[-2]
						p2 = collect_poly_points[-1]

						print "draw line --> p1, p2: ({}, {})".format(p1, p2)
						self.draw_polyline(
							[p1, p2], (255, 0, 0))
						self.update()          


		return QtGui.QLabel.event(self, event)


	def mouseMoveEvent(self, event):
		pos = event.pos()
		if event.buttons() == QtCore.Qt.LeftButton:
			edit_method = self.parent().edit_method
			if edit_method == 'segmentation':
				self.collect_points.append((np.int(pos.x()), np.int(pos.y())))

			# else:
			# 	if(edit_method == 'polygon'):
			# 		self.collect_poly_points.append([np.int(pos.x()), np.int(pos.y())])

		elif event.buttons() == QtCore.Qt.RightButton:
			pass

		if event.buttons() == QtCore.Qt.NoButton and self.parent().edit_method == "polygon":
			x = pos.x()
			y = pos.y()
			if self.ImgLoaded == False or x < 0 or x >= self.w or y < 0 or y >= self.h:
				# print "!!! mouse cursor out of img area {}".format((x, y))
				pass
			else:
				self.visualP = [y, x]


	def mouseDoubleClickEvent(self, event):
		if self.ImgLoaded == False or self.parent().mode != 'edit':
			return

		if event.buttons() == QtCore.Qt.LeftButton:
			if len(self.collect_poly_points) >= 2 or len(self.zoomed_collect_points) > 0:
				if self.parent().edit_method == "polygon":
					self.update_polygon_value()
					self.update_disp()

			self.update()		

	def disp_ori_on(self):
		print "displaying original img ON"
		## unable edit
		self.parent().mode = "view"
		img_arr = self.ref_pic.img_arr
		self.img_arr_tmp = self.img_arr.copy()

		self.img_arr = img_arr
		self.update()	


	def disp_segBI_on(self):
		print "displaying segBI ON"
		## unable edit
		self.parent().mode = "view"
		self.img_arr_tmp = self.img_arr.copy()
		img_arr = self.ori_img.copy()

		## display binary img
		segBI = np.zeros(img_arr.shape[:2], np.uint8)
		segBI[self.seg_arr == self.current_label] = 255
		segBI = cv2.cvtColor(segBI, cv2.COLOR_GRAY2RGB)


		if self.Zoomed == True:
			large_segBI = Image.fromarray(segBI).resize((self.w * self.zRate, self.h * self.zRate), Image.NEAREST)
			cropped_segBI = large_segBI.crop(tuple(self.zoom_pos))
			segBI = np.array(cropped_segBI)

		self.img_arr = segBI
		self.update()


	def disp_segBI_off(self):
		print "displaying segBI OFF"
		## enable edit
		self.parent().mode = "edit"

		self.img_arr = self.img_arr_tmp.copy()
		self.update()	


	def zoom_in(self):
		"""
		- resize and crop cv2_image according to visualP position
		- resize and crop zoomed_BI with same zRate and position
		- replace self.cv2_image with zoomed image
		"""
		if self.Zoomed == True:
			return
		self.Zoomed = True

		W = self.w * self.zRate
		H = self.h * self.zRate
		if self.large_ref_img is None:
			ori_sized_img = Image.fromarray(self.ori_sized_img_arr)
			self.large_ref_img = np.array(ori_sized_img.resize((W, H), Image.NEAREST))

		# crop large_img based on current mouse position -- visualP
		y, x = self.visualP
		left = (self.zRate - 1) * x
		upper = (self.zRate - 1) * y
		right = self.zRate * x + (self.w - x)
		lower = self.zRate * y + (self.h - y)
		self.zoom_pos = [left, upper, right, lower]

		# crop on zooming position
		self.ref_pic.img_arr = np.array(Image.fromarray(self.large_ref_img).crop(tuple(self.zoom_pos)))

		self.img_arr = np.array(self.large_ref_img) 	 # clear img_arr (uncropped)
		self.update_disp()


		## 2. draw polyline in self.collect_poly_points
		if len(self.collect_poly_points) > 1:
			large_points = self._point_small2large(self.collect_poly_points)  # project points on to large img
			# large_points = self._large_point_inbound(self.collect_poly_points, self.break_i)  # transform all points in bound
			self.draw_polyline(
				large_points, (255, 0, 0))			

		print ">>>>>>>>> len(zoomed_collect_points): ", len(self.zoomed_collect_points) 
		if len(self.zoomed_collect_points) == 0 and len(self.collect_poly_points) != 0:
			# print "############## zoom in ... push in point: ", self.collect_poly_points[-1]
			point_large = self._point_small2large([self.collect_poly_points[-1]])[0]
			# print "############## zoom in ... large point: ", point_large 
			self.zoomed_collect_points.append(point_large)
			self.i_mark = len(self.collect_poly_points)-1
			# print ">>>>>>>>> zoomed_collect_points: {}".format(self.zoomed_collect_points)

		self.update()   # display img_arr


	def zoom_out(self):
		if self.Zoomed == False:
			return

		self.Zoomed = False

		# 1. Set display img (cv2_image) as original sized image
		self.ref_pic.img_arr = self.ori_img.copy()

		self.i_mark = 0

		if self.parent().edit_method == "polygon":	
			self.zoomed_BI = None

		for i, point in enumerate(self.zoomed_collect_points):
			if i == 0 and len(self.collect_poly_points) > 0:
				print "skip 1st point in zoomed_collect_points"
				continue
			else:
				pointS = self._point_large2small([point])
				# print "pointS = ", pointS				
				self.collect_poly_points.extend(pointS)

		print "collected points: ", self.collect_poly_points
		## 1. draw all lines in line_set / _update_disp_line2
		self.update_disp()
		## 2. draw all lines in collect_poly_points
		if len(self.collect_poly_points) > 1:
			# print "draw collect_poly_points --> {}".format(self.collect_poly_points)
			self.draw_polyline(
				self.collect_poly_points, (255, 0, 0))
		self.zoomed_collect_points = []

		self.update()


	def draw_points(self, points, Positive=True):
		if len(points) == 0:
			print "No points to draw!!! (Positive -- ", Positive, ")"
			return

		if Positive == True:
			color = (0, 255, 0, 180)    # Green for posi
		else:
			color = (255, 0, 0, 180)    # Red for neg

		if (self.img_arr is not None):
			img = Image.fromarray(self.img_arr)
			draw = ImageDraw.Draw(img, mode='RGBA')
			draw.point(points, fill=color)

			self.img_arr = np.array(img)


	def p2p_distance(self, p1, p2):
		# calculate Euclidean dist between 2 points
		return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**(1 / 2.0)


	def draw_polyline(self, line_points, color=(255, 0, 0)):
		"""
		Draw polyline
		@line: [(p1x, p1y), (p2x, p2y), ....]
		@highlight: True/False
		"""
		# print ">>>>>>>>>>> ", t
		for p in line_points:
			if self._point_outofbound(p) == True and self.Zoomed == True:
				line_points, breaki = self._large_point_inbound(line_points, [])
				break

		# print "draw points: {}".format(line_points)

		if (self.img_arr is not None) and (len(line_points) > 0):
			img = Image.fromarray(self.img_arr)
			draw = ImageDraw.Draw(img, mode='RGB')

			j = 0
			for i in range(len(line_points) - 1):
				# default width
				wi = 2
				# print "color: ", color
				p1 = line_points[i]
				p2 = line_points[i + 1]

				draw.line([p1[0], p1[1], p2[0], p2[1]], fill=color, width=wi)

			self.img_arr = np.array(img)

	 

	def annotate_superpixel(self):
		if not (self.seg_index is None) and self.collect_points:
			ncol = self.seg_index.shape[1]
			# y -- row, x -- col; list of index of all collect_points in pic
			# (flattened vec)
			index = [y * ncol + x for x, y in self.collect_points]
			superpixel_set = set(self.seg_index.ravel()[index])
			selected_index = [
				i for i, x in enumerate(
					self.seg_index.ravel()) if x in superpixel_set]

			self.update_segvalue(selected_index, self.current_label)

	def update_segvalue(self, selected_index, label):
		"""
				1) mark selected_index with current label
				1) update img_arr according to seg_disp and label
				2) mark_boundaries on img_arr
				3) reset collect_points
		"""
		self.seg_arr.ravel()[
			selected_index] = label  # 1) mark selected_index with current label

		# tmp = self.seg_disp.reshape((-1, 3))  # 3 cols: R, G, B
		# # different color represents diff labels
		# tmp[selected_index, :] = self.color_map[label]
		# self.seg_disp = tmp.reshape(self.seg_disp.shape)

		# self.img_arr = np.array(self.ref_pic.img_arr *
		# 						(1 -
		# 						 self.alpha) +
		# 						self.alpha *
		# 						self.seg_disp, dtype=np.uint8)

		self.update_segDisp_from_segArr()

		self.img_arr = np.array(
			self.ref_pic.img_arr * (1 - self.alpha) + self.alpha * self.seg_disp_polygon, dtype=np.uint8)

		self.img_arr = np.array(
			mark_boundaries(
				self.img_arr,
				self.seg_index) * 255,
			dtype=np.uint8)

		self.collect_points = []  # reset collect_points[]
		self.update()


	def update_polygon_value(self):
		"""
				1) draw polygon according to seg_arr and label (color)
				2）reset seg_arr, collect_poly_points
		"""
		label = self.current_label
		if (not self.seg_arr is None):
			if self.Zoomed == True:
				# print ">>>>>> [update_polygon] len(self.collect_poly_points): ", len(self.collect_poly_points)
				# print ">>>>>> [update_polygon] self.zoomed_collect_points: ", self.zoomed_collect_points
				for i, point in enumerate(self.zoomed_collect_points):
					if i == 0 and len(self.collect_poly_points) > 0:
						print "skip 1st point in zoomed_collect_points"
						continue
					else:
						pointS = self._point_large2small([point])
						print "pointS = ", pointS				
						self.collect_poly_points.extend(pointS)
				
				# print ">>>>>> [update_polygon] self.collect_poly_points: ", self.collect_poly_points

			## draw on img_arr
			img = Image.fromarray(self.seg_arr)
			draw = ImageDraw.Draw(img)
			tu_collect_points = [tuple(p) for p in self.collect_poly_points]
			draw.polygon(tu_collect_points, fill=label, outline=label)
			self.seg_arr = np.array(img)

			## draw on seg_disp
			# self.update_segDisp_from_segArr()

			self.collect_poly_points = []
			if self.Zoomed == True:
				self.zoomed_collect_points = []
			self.update_disp()
			self.update()


	def set_ref_img(self, img_arr, img_path):
		# seg_arr (r * c) 2D size, init with 255
		# seg_arr[r, c] -- label of pixel on (r, c)
		print ">>>>>>>>>>> img_arr.shape: ", img_arr.shape
		if self.ref_pic:
			## resize to 720P if original image larger than 720
			self.h0 = img_arr.shape[0]
			self.w0 = img_arr.shape[1]
			self.ori_sized_img_arr = img_arr.copy()
			if (self.h0 > 720):
				# img720 = Image.fromarray(img_arr)
				# shape720 = (int(self.w0 * (720.0/self.h0)), 720)
				# print ">>>>>>> shape720: ", shape720
				# img720.show()
				img_arr = np.array(Image.fromarray(img_arr).resize((int(self.w0 * (720.0/self.h0)), 720), Image.LANCZOS))


			self.skip = False   # skip init to False  
			self.large_ref_img = None
			self.ImgLoaded = True
			self.ref_pic.img_arr = img_arr
			# print ">>>>>>>>>> self.ref_pic.img_arr.shape: ", self.ref_pic.img_arr.shape
			# self.lineSeg_BI_dict = {}
			self.ori_img = self.ref_pic.img_arr.copy()

			if self.parent().edit_method == "polygon" and (self.seg_disp_polygon is None or len(self.seg_disp_polygon.shape) < 3):
				self.seg_disp_polygon = self.ref_pic.img_arr.copy()
			self.seg_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)
			self.seg_arr[:] = 255
			self.ref_pic.img_path = img_path
			self.w = self.ref_pic.img_arr.shape[1]
			self.h = self.ref_pic.img_arr.shape[0]

			self.update_disp()
			return True
		return False
		

	def resize_img_arr(self, img_arr, newW, newH, RESAMPLE=PIL.Image.NEAREST):
		# return cv2.resize(img_arr, (newW, newH))
		if len(img_arr.shape) == 2:
			Img = Image.fromarray(img_arr, 'I')
		else:
			Img = Image.fromarray(img_arr)
		Img = Img.resize((newW, newH), RESAMPLE)
		return np.array(Img)


	def do_segmentation(self):
		# Apply segmentation and update the display
		"""
		1) Do pre-segmentation (2 methods available)
		2) seg_index -- init with pre-segmentation result
		3) seg_disp -- each pixel is colored according to its class label
		4.1) img_arr -- on ori_img, color each pixel according to seg_disp
		4.2) img_arr -- mark boundaries of segmentations according to seg_index
		"""
		print "## run [do_segmentation]..."
		## Apply segmentation and update the display
		ori_img = self.ref_pic.img_arr
		ori_img_path = self.ref_pic.img_path
		ori_img_tmp = ori_img.copy()
		# print "ori_img.shape: ", ori_img.shape
		ori_h = ori_img.shape[0]
		ori_w = ori_img.shape[1]

		# if image larger than 180p, run segmentation on resized img
		if ori_h > 720:
			smaller_h = 720
			smaller_w = int((200.0/ori_h)*ori_w)

			ori_img = self.resize_img_arr(ori_img, smaller_w, smaller_h, PIL.Image.LANCZOS)

		print "Please Click anywhere to start"
		logging.warn(u'请点击图片以开始 Please Click anywhere to start\n')
		
		# print "    ori_img.shape: ", ori_img.shape
		sp = self.seg_params
		if self.seg_method == 'slic':
			n_segments, compactness, sigma = np.int(sp[0]), np.int(sp[1]), sp[2]
			self.seg_index = slic(ori_img, n_segments=n_segments, compactness = compactness,
								  sigma = sigma)
		elif self.seg_method == 'felzenszwalb':
			scale, min_size, sigma = np.int(sp[0]), np.int(sp[1]), sp[2]
			self.seg_index = felzenszwalb(ori_img, scale=scale, min_size = min_size,
										  sigma = sigma)
		elif self.seg_method == 'contour':
			thesh = sp[0]
			mask_dir = os.path.split(ori_img_path)[0]
			mask_name = os.path.splitext(os.path.basename(ori_img_path))[0] + '_mask'
			seg_name = os.path.splitext(os.path.basename(ori_img_path))[0] + '_seg'
			mask_path = '{}\{}.mat'.format(mask_dir, mask_name)			
			seg_path = '{}\{}.png'.format(mask_dir, seg_name)
			mask = sio.loadmat(mask_path)['mask']
			mask = self.resize_img_arr(mask, 1280, 720)
			
			self.seg_index = cv2.imread(seg_path)[:,:,0]
			#self.seg_index = cv2.resize(self.seg_index, (1280, 720))
			self.seg_index[mask < thesh] = -1
			

		#self.seg_index = np.asarray(self.seg_index, np.int32)
		# print "    type(seg_index[0][0]): ", type(self.seg_index[0][0])
		r = ori_img.shape[0]
		c = ori_img.shape[1]
		if (self.seg_index.shape[0] != ori_h or self.seg_index.shape[1] != ori_w):
			self.seg_index = self.resize_img_arr(self.seg_index, ori_w, ori_h)
			ori_img = ori_img_tmp        

		self.seg_disp = self.color_map[self.seg_arr.ravel()].reshape((ori_h, ori_w,3))
		# self.img_arr = np.array(ori_img * (1 - self.alpha)  + self.alpha * self.seg_disp,dtype=np.uint8)
		#self.img_arr = np.array(mark_boundaries(self.img_arr, self.seg_index) * 255,dtype=np.uint8)
		self.img_arr = mark_boundaries(self.img_arr, self.seg_index)

		self.update()


	def load_segimage(self, seg_path):
		# print "@@@@@@@@@@@@@@@@@@@@@@@@@@ loading_segimage"
		# load and display a segmented img
		# 1) seg_path -- e.g. 00001_seg.png  --loads to--> self.seg_arr
		# 2） load 0001_seg.json 'lines' --> self.line_dict

		# 1) load Segmentation
		seg = np.array(Image.open(seg_path))
		# seg_arr is 2D r*c
		assert(len(seg.shape) == 2)
		self.seg_arr = seg

		# print set(self.seg_arr.ravel()), 'seg arr sum'
		assert(not (self.ref_pic is None))
		ori_img = self.ref_pic.img_arr
		# if ori_img.shape[:2] != self.seg_arr.shape[:2]:
		assert(ori_img.shape[:2] == self.seg_arr.shape[:2])

		_, ext = os.path.splitext(seg_path)
		seg_meta_path = seg_path.replace(ext, '.json')

		# print "seg_meta_path: ", seg_meta_path

		if os.path.exists(seg_meta_path):
			print "Loading seg.png and .json file from {}...".format(seg_meta_path)
			d = read_json(seg_meta_path)
			print "read_json: ", seg_meta_path
			if ('polylines' in d.keys()):
				tmp_list = d['polylines']
				for tmp in tmp_list:
					loadline = self.LaneLine()
					loadline.line_type = tmp['line_type']
					points = tmp['points']
					for p in points:
						loadline.points.append([int(p[0]), int(p[1])])
					# loadline.break_idx[:] = tmp['break_idx']
					break_i = tmp['break_idx']
					for i in break_i: 
						loadline.break_idx.append(int(i))
					self.line_set.add(loadline)
				print self.line_set

			if ('binarized_line' in d.keys()):
				biLine_dict = d['binarized_line']
				if "label_type" in biLine_dict.keys():
					labelType_dict = biLine_dict['label_type']
					for label in labelType_dict.keys():
						t = labelType_dict[label]
						# print ">>>>>>>>>>>> ",t
						# update lineSegDist_dict (keep Disp sync with final_BI)
						# self.lineSegDisp_dict[t] = self._get_lineSeg_Disp(t)

				if "ID_label" in biLine_dict.keys():
					for str_ID in biLine_dict['ID_label'].keys():
						ID = int(str_ID)
						if type(biLine_dict['ID_label'][str_ID]) == list:
							label = [int(biLine_dict['ID_label'][str_ID][0])]   # wrap label in a list
						else:
							label = [int(biLine_dict['ID_label'][str_ID])]
						self.line_ID2label[ID] = label

				if "LaneLineID_png_offset" in biLine_dict.keys():
					self.int8_to_uint8_OFFSET = int(biLine_dict["LaneLineID_png_offset"])
				else:
					self.int8_to_uint8_OFFSET = 128

			if ('skip' in d.keys()):
				flag = int(d['skip'])
				self.skip = True if flag == 1 else False
				print ">>> thisImage.skip = {}".format(self.skip)
			else:
				self.skip = False

		self.update_disp()
		self.update()


	def reset_current_superpixel(self, x, y):
		# reset superpixel (x, y) to ignore_label

		ncol = self.seg_index.shape[1]
		idx = self.seg_index.ravel()[y * ncol + x]
		selected_index = [
			i for i, v in enumerate(
				self.seg_index.ravel()) if v == idx]
		# print selected_index, '==============='
		self.update_segvalue(selected_index, self.ignore_label)


	def save(self, save_path):
		# print "*@@@@@@@@@@@@@saving@@@@@@@@@@@@@"
		# save_path: "{}/{}_seg.png".format(self.anno_dir, base_name)
		# ## Check if illegal
		# ID, legal = self.check_ID2label_llegal()
		# if legal == False:
		# 	self.show_conflict_msgBox(ID)
		# 	return False
		# 1) save seg_arr into .png file
		Image.fromarray(np.array(self.seg_arr, dtype=np.uint8)).save(save_path)

		# # 2) save final_BI into xxxx_LaneLineBI.png, final_ID into xxxx_LaneLineID.png file
		# lineBI_save_path = save_path[:-8] + "_LaneLineBI.png"
		# lineID_save_path = save_path[:-8] + "_LaneLineID.png"
		# lineBI_arr = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
		# for t in sorted(self.lineSeg_BI_dict.iterkeys()):
		# final_BI = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
		# final_BI[self.final_BI != 0] = 255
		# Image.fromarray(final_BI).show()
		# Image.fromarray(self.final_BI).save(lineBI_save_path)
		# lineID_arr = int8_to_uint8(self.final_ID, self.int8_to_uint8_OFFSET)
		# Image.fromarray(lineID_arr).save(lineID_save_path)

		# 3) save self.label_type and self.label_names info into json
		_, ext = os.path.splitext(save_path)
		save_meta_path = save_path.replace(ext, '.json')
		self.save_meta(save_meta_path)

		return True


	def save_meta(self, save_path):
		d = OrderedDict()
		if self.parent().edit_method == "segmentation" or self.parent().edit_method == "polygon":
			d['label_type'] = self.label_type
			d['label_names'] = self.label_names
			# print "want to save: ", [line.__dict__ for line in self.line_set]

		# ## save polyline data
		# print ">>>>>>>>>>line_set --> {}".format(self.line_set)
		# d['polylines'] = [line.__dict__ for line in self.line_set]

		# ## save binarized_line data
		# # d = dict()
		# biLine_dict = dict()
		# d["binarized_line"] = biLine_dict

		# ## 1) save label-type key-value paires in field "label_type"
		# labelType_dict = dict()
		# biLine_dict["label_type"] = labelType_dict

		# for t in self.line_types.keys():
		# 	if "label" not in self.line_types[t].keys():
		# 		continue
		# 	label = self.line_types[t]["label"]
		# 	if label in self.final_BI:
		# 		labelType_dict[label] = t
		# # print ">>>>>>>>>>>>>>> ", biLine_dict

		# biLine_dict["label_type"] = labelType_dict

		# ## 2) save ID-label key-value paires in field "ID_label"
		# biLine_dict["ID_label"] = self.line_ID2label
		# biLine_dict["LaneLineID_png_offset"] = self.int8_to_uint8_OFFSET

		# d["binarized_line"] = biLine_dict

		if np.count_nonzero(self.seg_arr) != 0:
			self.skip = False
		d["skip"] = 1 if self.skip == True else 0

		write_json(save_path, d)		


	def checkpoint(self):
		"""
		save current segmentation state
		seg method:
		seg parameter:
		index_map
		"""
		pass