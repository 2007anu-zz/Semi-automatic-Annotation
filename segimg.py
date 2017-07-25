#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
For LaneLine
"""

import os
import sys

from PyQt4 import QtGui
from PyQt4.QtGui import *
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import time
from collections import OrderedDict
import operator
from PIL import ImageFont
import cv2

from skimage.segmentation import felzenszwalb,slic
from skimage.segmentation import mark_boundaries
from skimage import measure
from mygrabcut import MyGrabCut
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


def int8_to_uint8(array_int8, offset):
	"""
	convert np.array int8 into np.array uint8
	for self.final_ID
	"""
	return (array_int8 + offset).astype(np.uint8)


def uint8_to_int8(array_uint8, offset):
	"""
	convert np.array uint8 into np.array int8
	for self.final_ID
	"""
	return (array_uint8 - offset).astype(np.int8)


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

class modifyTypeID_Dialog(QDialog):
	def __init__(self, typeList, curType, curID, parent = None):
		super(modifyTypeID_Dialog, self).__init__(parent)

		self.select_type = typeList[0]
		self.select_ID = curID 

		Vlayout = QtGui.QVBoxLayout(self)

		# widget for editing the date
		self.cb_label = QtGui.QLabel()
		self.cb_label.setText("LaneLine Type: ")
		self.sp_label = QtGui.QLabel()
		self.sp_label.setText("LaneLine ID: ")

		self.cb = QtGui.QComboBox()
		self.cb.addItems(typeList)
		self.cb.setCurrentIndex(typeList.index(curType))
		self.cb.currentIndexChanged.connect(self.type_change)

		self.sp = QtGui.QSpinBox()
		self.sp.setRange(-128, 127)
		self.sp.setValue(curID)
		self.sp.valueChanged.connect(self.ID_change)

		# add widgets to layout
		Hlayout1 = QtGui.QHBoxLayout(self)
		Hlayout1.addWidget(self.cb_label)
		Hlayout1.addWidget(self.sp_label)
		Vlayout.addLayout(Hlayout1)
			
		Hlayout2 = QtGui.QHBoxLayout(self)
		Hlayout2.addWidget(self.cb)
		Hlayout2.addWidget(self.sp)
		Vlayout.addLayout(Hlayout2)

		# OK and Cancel buttons
		buttons = QDialogButtonBox(
			QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
			Qt.Horizontal, self)
		buttons.accepted.connect(self.accept)
		buttons.rejected.connect(self.reject)
		Vlayout.addWidget(buttons)

	def type_change(self, idx):
		self.select_type = self.cb.itemText(idx)

	def ID_change(self):
		self.select_ID = self.sp.value()

	# get selected Type and ID from the dialog
	def get_type_and_ID(self):
			return (self.select_type, self.select_ID)
			
	# static method to create the dialog and return (type, ID, accepted)
	@staticmethod
	def getTypeAndID(typeList, curType, curID, parent = None):
		dialog = modifyTypeID_Dialog(typeList, curType, curID, parent)
		dialog.setWindowTitle("Select line type and ID")
		result = dialog.exec_()
		select_type, select_ID = dialog.get_type_and_ID()
		return (select_type, select_ID, result == QDialog.Accepted)


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
		self.ref_pic = None
		self.color_map = None
		self.label_names = None
		self.alpha = 0.35
		self.current_label = 0
		self.seg_params = None
		self.ignore_label = 255

		# line type and its color on annotated img (...TBE... should read
		# through config json file)
		self.line_types = OrderedDict()
		# self.line_types = {}
		self.normal = {}
		self.highlight = {}
		self.cutline = {}

		# self.line_width = {'polygon': 2, 'laneline': 2, 'highlight_line': 5}

		# self.line_info = OrderedDict([
		# 	('green_line', (46,204,113)),	# green
		# 	('blue_line', (46,58,204)),	# blue
		# 	('pink_line', (204,46,137)),	# pink
		# 	('yellow_line', (204,192,46))	# yellow
		# ])
		self.load_line_info()
		# cur line type selected by ratiobtn
		# self.cur_line_type = self.line_types.keys()[0]
		self.cur_line_type = "fnc_virtualLine"
		self.cur_line_type2 = self.line_types.keys()[0]
		self.cur_line_label = self.line_types[self.cur_line_type2]['label']

		## ID for laneLineSeg; ..., -2, -1, 0, 1, 2, 3, ...
		self.cur_line_ID = 0 

		# select line Method 2. buffer pixel to check if point is near a line
		self.mouse_epsilon = 6

		self.cur_line = None

		self.allow_cutline = False  # Only allow cutline when already drawn actual line before
		# True if the tinyline we're drawing now is a cutline (c + mouseLeftBtn
		# setting cutline's end point)
		self.cutline = False
		self.break_i = []  # index of break points on cur_line
		self.selected_line = None  # line selected by G + mouseLeftBtn click on img

		self.line_set = None  # all Lane_Lines in pic


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

		# cursor position saved (y, x)
		self.cursorP = []
		self.collect = None
		self.V_Edge = False

		# key -- line_label; Value -- [] of 2; 
		# value[0] -- BIBlock_stk; value[1] -- final_BI (0 or label) binary image 

		self.grabcut = None

		# self.lineSeg_Disp = None
		self.lineSegDisp_dict = {}   # dict for all line_type's seg_disp arr (painted with its corresponding color_fill)
		self.ImgLoaded = False  # True if image has already been loaded from folder
		self.ori_image = None  	# orignal sized img (for zoom out)


		######## 1_6 version staff ##############
		self.bbox = []  # [(TLx, TLy), (BRx, BRy)]
		self.posi_dot_size = 4
		self.neg_dot_size = 4
		self.pos_px = set()
		self.neg_px = set()

		self.cur_lineSeg_BI = None     # numpy binary array (row, col); lineSeg for current op; top of lineSeg_BI_stk; (0 or 255)
		# key -- line_label; Value -- [] of 2; 
		# value[0] -- BIBlock_stk; value[1] -- final_BI (0 or label) binary image
		# self.lineSeg_BI_dict = {}
		# bitmap of lineTypeLabel; [0, 255] np.uint8
		self.final_BI = None
		# bitmap of ID (-128 as background); [-128, 127] np.int8; keep sync with final_BI (has same nonzero elements)
		# when saving into png, scale to np.uint8 using int8_to_uint8; when loading from png, scale using uint8_to_int8 
		self.final_ID = None
		self.int8_to_uint8_OFFSET = 128   # background of final_ID is -self.int8_to_uint8_OFFSET

		## key: line_ID; value: line_Label of that ID
		self.line_ID2label = {}

		self.BiBlock_stk = []
		self.UndoCacheMaxSize = 8
		self.img_arr_tmp = None

		self.MannualMode = True 

	def set_reference_pic(self, ref_pic):
		self.ref_pic = ref_pic

		# seg_arr init to ignore_label
		if not (ref_pic.img_arr is None):
			self.seg_arr = np.zeros(ref_pic.img_arr.shape[:2], dtype=np.uint8)
			self.seg_arr[:] = self.ignore_label
			# init line_dict
			self.line_set = set()


	def set_seg_params(self, seg_params):
		self.seg_params = seg_params

	def init_seg(self):
		ori_img = self.ref_pic.img_arr
		self.seg_arr = np.zeros(ori_img.shape[:2], dtype=np.uint8)
		self.seg_arr[:] = self.ignore_label
		# init line_dict
		self.line_set = set()


	def _get_snap_point(self, mouseP, rectH, rectW):
		"""
		Automatically get the middle/edge point of a line based on Canny Edge detect
		- for [cursor snap to middle]
		- if self.line2_mode = "polygonMinus" or "polygonPlus": snap to edge
		return: (y, x)
		"""
		# print "Getting mid point {}...".format(mouseP)
		## Crop img
		self.edge_x = [] 	# x of edge points which are on the same row with mouseP
		self.edge_y = []	# y of edge points which are on the same col with mouseP

		if self.Zoomed == True:
			rectH = self.h0/2
			rectW = self.w0/2

		TLy = mouseP[0] - rectH/2
		TLx = mouseP[1] - rectW/2
		if TLy <= 0:
			TLy = 0
		if TLy >= self.h:
			TLy = self.h - 1
		if TLx <= 0:
			TLx = 0
		if TLx >= self.w:
			TLx = self.w - 1
		TL = (TLy, TLx)

		BRx = mouseP[1] + rectW/2
		if BRx >= self.w:
			BRx = self.w - 1

		# print "TL: ", TL
		# print "rect size = ({}, {})".format(rectH, rectW)
		# print self.cv2_image.shape
		img_rect = self.ref_pic.img_arr[TL[0] : TL[0]+rectH, TL[1] : TL[1]+rectW]
		# print img_rect.shape

		## Canny Edge
		gaussian_kernel = 5
		if self.Zoomed == True:
			gaussian_kernel = 1
		img_g = cv2.cvtColor(img_rect, cv2.COLOR_BGR2GRAY)
		img_gn = cv2.GaussianBlur(img_g, (gaussian_kernel, gaussian_kernel), 0)

		canny_high_threshold, th_otsu2 = cv2.threshold(img_gn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		canny_low_threshold = 0.5 * canny_high_threshold

		img_gnc = cv2.Canny(img_gn, canny_low_threshold, canny_high_threshold)

		# mouse pointed row (in local rect)
		row = img_gnc[mouseP[0] - TL[0]]
		mouseX = mouseP[1] - TL[1]

		# mouse pointed col (in local rect)
		col = img_gnc[:, mouseX]
		mouseY = mouseP[0] - TL[0]

		# get edge_x
		x1, x2 = self._get2nearest(row, mouseX)
		self.edge_x = [x1, x2]
		# get edge_y
		y1, y2 = self._get2nearest(col, mouseY)
		self.edge_y = [y1, y2]

		# if x1 - x2 == 0 and y1 - y2 == 0:
		# 	# print ">> no edge detected at local rect!"
		# 	return mouseP
		# elif x1 - x2 == 0 or abs(x1 - x2) > abs(y1 - y2):
		# 	self.V_Edge = True  # edge is vertical --> use edge_x
		# 	return (TL[0] + (y1 + y2)/2, mouseP[1])
		# elif y1 - y2 == 0 or abs(x1 - x2) < abs(y1 - y2):
		# 	self.V_Edge = False  # edge is horizon --> use edge_y
		# 	return (mouseP[0], TL[1] + (x1 + x2)/2)

		if self.parent().edit_method == "line" or (self.parent().edit_method == "line2" and self.parent().line2_mode == "smartGC"):
			## [smartGC] mode --> snap to middle point
			if x1 - x2 != 0:
				return (mouseP[0], TL[1] + (x1 + x2)/2)
		elif self.parent().edit_method == "line2" and (self.parent().line2_mode == "polygonMinus" or self.parent().line2_mode == "polygonPlus"):
			## [polygonMinus] or [polygonPlus] mode --> snap to edge
			if abs(x1 - mouseP[1]) <= abs(x2 - mouseP[1]):
				# print "aaa"
				return (mouseP[0], TL[1] + x1)
			else:
				# print "bbb"
				return (mouseP[0], TL[1] + x2)   

		# print "ccc"
		return mouseP
	
		# pMid = (mouseP[0], (nearest + nearest2)/2 + TL[1])

		# # get points on edge of cur row
		# # if no edge detected, use rect bound
		# self.edge_x.append(min(nearest2, nearest))
		# self.edge_x.append(max(nearest2, nearest))
		# self.edge_x[0] = TLx if self.edge_x[0] == mouseX else self.edge_x[0]
		# self.edge_x[1] = BRx if self.edge_x[1] == mouseX else self.edge_x[1]

		# return pMid


	def _get2nearest(self, arr, mouseP):
		"""
		Get 2 nearest point's x coordinates, if exist
		If no edge detected, return mouseP
		"""
		white0 = np.where(arr == 255)[0]
		len_tmp = len(white0)
		white = []
		# element (57, 58, 59, 62) --> (57, 62)
		if (len_tmp > 0):
			white.append(white0[0])
		for i in range(1, len_tmp):
			if (white0[i-1] + 1 == white0[i]):
				continue
			white.append(white0[i])
				
		# print white
		i = 0	# white[i] is the 1st element >= mouseX
		for i in range(len(white)):
			if (white[i] >= mouseP):
				break
	
		nearest = mouseP
		nearest2 = mouseP
		
		if len(white) <= 1:
			# return mouseP if no white points exists in this row
			pass
		elif i == 0:
			nearest = white[i]
			nearest2 = white[i+1]
		elif i == len(white):
			nearest = white[i-1]
			nearest2 = white[i-2]
		elif white[i] == mouseP and i < len(white)-1:
			nearest = mouseP
			nearest2 = min(abs(white[i-1]-mouseP), abs(white[i+1]-mouseP))
		else:
			nearest = white[i]
			nearest2 = white[i-1]

		return (nearest, nearest2)


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
			if (self.parent().edit_method == "segmentation"):
				r, c = self.seg_arr.shape[0], self.seg_arr.shape[1]
				self.seg_disp = self.color_map[self.seg_arr.ravel(), :].reshape(
					(r, c, 3))
				self.img_arr = np.array(
					ref_pic.img_arr * (1 - self.alpha) + self.alpha * self.seg_disp, dtype=np.uint8)

			# 2) draw lines on img_arr
			if (self.parent().edit_method == "line") and (not (self.line_set is None)):
				self.img_arr = ref_pic.img_arr
				# print "Zoomed: ", self.Zoomed 
				for line in self.line_set:
					if self.Zoomed == True:
						points = self._point_small2large(line.points)
						print "disp large points -- {}".format(points)
						# large_points, break_i = self._large_point_inbound(large_points, line.break_idx)
					else:
						points = line.points
					self.draw_polyline(						
							points, line.break_idx, line.line_type, False)

			# 3) draw lines on img_arr
			if (self.parent().edit_method == "line2"):
				self._update_disp_line2()


	def _update_disp_line2(self):
		print "Update Disp for line2..."
		ori_img = self.ref_pic.img_arr  # numpy array

		if not (ori_img is None):
			self.img_arr = ori_img
			## display all lineType
			lineSeg_Disp = self._get_lineSeg_Disp_all()
			# if self.cur_line_type2 not in self.lineSegDisp_dict.keys() and self.cur_line_type2 != "all":
			# 	lineSeg_Disp = np.zeros(self.img_arr.shape, np.uint8)
			# 	lineSeg_Disp[:] = 255
			# 	# print ">>>>> 1 type lineSeg_Disp: ", type(lineSeg_Disp)
			# else:
			# 	lineSeg_Disp = self._get_lineSeg_Disp(self.cur_line_type2)
			# 	# print ">>>>> 2 type lineSeg_Disp: ", type(lineSeg_Disp)
			# 	# lineSeg_Disp = self.lineSegDisp_dict[self.cur_line_type2]
			## points for drawing line_ID curve
			pointsXY, IDtext = self.get_IDCurvePoints()

			if self.Zoomed == True:
				## zoom (resize and crop) lineSeg_Disp
				points_draw = []
				for XY in pointsXY:
					points_draw.append(self._point_small2large(XY))

				img_lineSegDisp = Image.fromarray(lineSeg_Disp)
				largeImg_lineSegDisp = img_lineSegDisp.resize((self.w * self.zRate, self.h * self.zRate), Image.NEAREST)
				cropped_lineSegDisp = largeImg_lineSegDisp.crop(tuple(self.zoom_pos))
				lineSeg_Disp = np.array(cropped_lineSegDisp)
			else:
				points_draw = pointsXY
			## visualize line ID
			lineSeg_Disp = self.draw_IDcurve(lineSeg_Disp, points_draw, IDtext)

			self.img_arr = np.array(
				ori_img * (1 - self.alpha) + self.alpha * lineSeg_Disp, dtype=np.uint8)

		# image = Image.fromarray(self.img_arr)
		# image.show()


	# def _update_disp_line2(self):
	# 	print "Update Disp for line2..."
	# 	ori_img = self.ref_pic.img_arr  # numpy array

	# 	if not (ori_img is None):
	# 		self.img_arr = ori_img
	# 		# 1) display final_BI on image for cur_line_type2
	# 		if (self.cur_line_type2 not in self.lineSeg_BI_dict.keys() and self.cur_line_type2 != "all"):
	# 			print "{} not in lineSeg_BI_dict.keys".format(self.cur_line_type2)
	# 			lineSeg_Disp = np.array(self.img_arr.shape)
	# 			lineSeg_Disp[:] = 255
	# 		else:
	# 			if self.cur_line_type2 != "all":
	# 				lineSeg_Disp = self.lineSegDisp_dict[self.cur_line_type2]
	# 				if self.Zoomed == True:
	# 					## zoom (resize and crop) lineSeg_Disp
	# 					img_lineSegDisp = Image.fromarray(lineSeg_Disp)
	# 					largeImg_lineSegDisp = img_lineSegDisp.resize((self.w * self.zRate, self.h * self.zRate), Image.NEAREST)
	# 					cropped_lineSegDisp = largeImg_lineSegDisp.crop(tuple(self.zoom_pos))
	# 					lineSeg_Disp = np.array(cropped_lineSegDisp)
	# 			else:
	# 				lineSeg_Disp = self._get_lineSeg_Disp_allType()  
	# 		self.img_arr = np.array(
	# 			ori_img * (1 - self.alpha) + self.alpha * lineSeg_Disp, dtype=np.uint8)

	# 	# image = Image.fromarray(self.img_arr)
	# 	# image.show()

	def get_IDCurvePoints(self):
		pointsXY = []
		IDtext = []
		bi_ID = np.zeros(self.final_BI.shape, np.uint8)
		for ID in self.line_ID2label.keys():
			XY = []
			bi_ID[:] = 0
			bi_ID[self.final_ID == ID] = 255

			if np.count_nonzero(bi_ID) == 0:
				continue
			px, py = self.get_bSpline_fit(bi_ID)
			for i in range(len(px)):
				XY.append([px[i], py[i]])
			pointsXY.append(XY)
			IDtext.append("ID" + str(ID))

		return (pointsXY, IDtext)


	def draw_IDcurve(self, img_arr, pointsXY, IDtext):
		"""
		draw on self.img_arr
		pointsXY: [[[x1, y1], [x2, y2], ...], [[x1, y1], [x2, y2], ...], ...]
		"""
		color = (0, 237, 35)
		font = ImageFont.truetype("/data/font/arialbd.ttf",25)
		img = Image.fromarray(img_arr)
		draw = ImageDraw.Draw(img, mode='RGBA')	

		for i, line in enumerate(pointsXY):
			if len(line) > 0:
				extBottom = line[0]
			else:
				continue
			for j in range(len(line) - 1):
				if line[j+1][1] > extBottom[1]:
					extBottom = line[j+1]
				draw.line((line[j][0], line[j][1], line[j+1][0], line[j+1][1]), fill=color + (200,), width=4)
			# print ">>>>>>>>> extBottom: {}; img_arr.h: {}".format(extBottom, img_arr.shape[0])
			if extBottom[1] + 25 > img_arr.shape[0]:
				extBottom[1] -= 25
			if extBottom[0] + 50 > img_arr.shape[1]:
				extBottom[0] -= 50
			if extBottom[0] - 50 < 0:
				extBottom[0] += 50
			draw.text(tuple(extBottom), IDtext[i], color, font=font)

		return np.array(img)


	def _get_lineSeg_Disp_all(self):
		# draw mask on img_arr with foreground painted in its "color_fill" (only display cur_line_type2 lineSeg)
		## always draw on original_sized arr
		img_arr = self.ori_img.copy()

		# self.final_BI = np.zeros(img_arr.shape, np.uint8) if self.final_BI is None else self.final_BI

		bi_t = np.zeros(self.final_BI.shape, np.uint8)
		# print "img_arr.shape = {}".format(img_arr.shape)

		# display all types in one img
		for lineType in self.line_types.keys():
			if "label" not in self.line_types[lineType].keys():
				continue
			# print ">>>>>>>>>>>>> ", lineType
			label = self.line_types[lineType]["label"]
			color_fill = self.line_types[lineType]["color_fill"][0]
			bi_t[:] = 0
			bi_t[self.final_BI == label] = 255
			### 1) visualize line type
			if np.count_nonzero(bi_t) > 0:
				# Image.fromarray(bi_t).show()
				bi_t_img = Image.fromarray(bi_t)
				img = Image.fromarray(img_arr)
				draw = ImageDraw.Draw(img, mode='RGB')
				draw.bitmap((0, 0), bi_t_img, fill=tuple(color_fill))
				img_arr = np.array(img)


		lineSeg_Disp = img_arr
		# Image.fromarray(lineSeg_Disp).show()
		return lineSeg_Disp


	def _get_lineSeg_Disp(self, t):
		# draw mask on img_arr with foreground painted in its "color_fill" (only display cur_line_type2 lineSeg)
		## always draw on original_sized arr
		img_arr = self.ori_img.copy()

		# self.final_BI = np.zeros(img_arr.shape, np.uint8) if self.final_BI is None else self.final_BI

		bi_t = np.zeros(self.final_BI.shape, np.uint8)
		# print "img_arr.shape = {}".format(img_arr.shape)

		if (t == "all"):
			# display all types in one img
			for lineType in self.line_types.keys():
				if "label" not in self.line_types[lineType].keys():
					continue
				# print ">>>>>>>>>>>>> ", lineType
				label = self.line_types[lineType]["label"]
				color_fill = self.line_types[lineType]["color_fill"][0]
				bi_t[:] = 0
				bi_t[self.final_BI == label] = 255
				### 1) visualize line type
				if np.count_nonzero(bi_t) > 0:
					# Image.fromarray(bi_t).show()
					bi_t_img = Image.fromarray(bi_t)
					img = Image.fromarray(img_arr)
					draw = ImageDraw.Draw(img, mode='RGB')
					draw.bitmap((0, 0), bi_t_img, fill=tuple(color_fill))
					img_arr = np.array(img)

		else:
			label = self.line_types[t]["label"]
			color_fill = self.line_types[t]["color_fill"][0]
			bi_t[:] = 0
			# print "bi_t.shape: {}; self.final_BI.shape: {}".format(bi_t.shape, self.final_BI.shape)
			bi_t[self.final_BI == label] = 255
			## 1) draw areas of line in t type as bitmap
			bi_t_img = Image.fromarray(bi_t)
			img = Image.fromarray(img_arr)
			draw = ImageDraw.Draw(img, mode='RGB')
			# draw.line([923, 395, 921, 472], fill = (0, 255, 0), width = 3)
			draw.bitmap((0, 0), bi_t_img, fill=tuple(color_fill))
			img_arr = np.array(img)

		## 2) visualize all line ID
		for ID in self.line_ID2label.keys():
			img_arr = self.draw_IDcurve(img_arr, ID)
			# img_arr = self.draw_groupRect(img_arr, ID)

			# ## 3) visualize selectedBlob if in select_line [mode]
			# if self.parent().select_line == True and len(self.selected_lineSeg_rect) != 0:
			# 	img_arr = np.array(img)
			# 	img_arr = self.draw_bbox(img_arr, self.selected_lineSeg_rect)
			# 	img = Image.fromarray(img_arr)

		lineSeg_Disp = img_arr
		# Image.fromarray(lineSeg_Disp).show()
		return lineSeg_Disp


	def get_bSpline_fit(self, bi_ID):
		# Image.fromarray(bi_ID).show()

		yx = np.where(bi_ID > 0)
		Y = yx[0].astype(float)
		X = yx[1].astype(float)
	
		norm_y = (Y - np.min(Y))/(np.max(Y) - np.min(Y))
	
		M = np.array([[-1,5,-10,10,-5,1], [5,-20,30,-20,5,0], [-10,30,-30,10,0,0], [10,-20,10,0,0,0], [-5,5,0,0,0,0], [1,0,0,0,0,0]])
		T = np.mat(np.c_[norm_y**5, norm_y**4, norm_y**3, norm_y**2, norm_y, np.ones(len(norm_y))])
		Q = np.mat(X).T
		# pinv = np.linalg.pinv(T*M)
		P = np.linalg.pinv(T * M) * Q
	
		down_y = np.max(Y)
		top_y = np.min(Y)
	
		y = np.array(range(int(top_y), int(down_y)))
		y = y.astype(float)
		ny = (y - np.min(y))/(np.max(y) - np.min(y))
		TT = np.mat(np.c_[ny**5, ny**4, ny**3, ny**2, ny, np.ones(len(ny))])
		x = TT*M*P
	
		return (x, y)		


	# def draw_IDcurve(self, img_arr, ID):
	# 	ID = int(ID)
	# 	img = Image.fromarray(img_arr)
	# 	draw = ImageDraw.Draw(img, mode='RGBA')	
	# 	bi_ID = bi_t = np.zeros(self.final_BI.shape, np.uint8)
	# 	bi_ID[self.final_ID == ID] = 255

	# 	if np.count_nonzero(bi_ID) == 0:
	# 		return img_arr

	# 	px, py = self.get_bSpline_fit(bi_ID)
	# 	for i in range(len(px)-1):
	# 		draw.line((px[i], py[i], px[i+1], py[i+1]), fill=(0, 237, 35, 200), width=4)

	# 	## find extreme bottom point
	# 	i, extBottom_y = max(enumerate(py), key=operator.itemgetter(1))
	# 	extBottom_x = px[i]
	# 	font = ImageFont.truetype("C:\WINDOWS\Fonts\SIMYOU.TTF", 18)
	# 	draw.text((extBottom_x, extBottom_y), "ID" + str(ID), (0, 237, 35), font=font)

	# 	return np.array(img)



	def draw_groupRect(self, img_arr, ID):
		ID = int(ID)
		img = Image.fromarray(img_arr)
		draw = ImageDraw.Draw(img, mode='RGBA')
		bi_ID = bi_t = np.zeros(self.final_BI.shape, np.uint8)
		bi_ID[self.final_ID == ID] = 255

		# Image.fromarray(bi_ID).show()

		## get contour -- cnt of polygon
		if self.parent().line2_mode == "polygonOtsu":
			bi_ID = cv2.GaussianBlur(bi_ID, (11, 11), 0)
			bi_ID = cv2.dilate(bi_ID, None, iterations=4)

		im2, contours, hierarchy = cv2.findContours(bi_ID, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		cnt_union = contours[0]
		for i, cnt in enumerate(contours):
			if i > 0:
				cnt_union = np.concatenate((cnt_union, cnt), axis=0)


		ellipse = cv2.fitEllipse(cnt_union)
		cv2.ellipse(img_arr,ellipse,(0,255,0,175),2)
		extBottom = tuple(cnt_union[cnt_union[:, :, 1].argmax()][0])

		font = cv2.FONT_HERSHEY_PLAIN
		cv2.putText(img_arr, "ID_"+str(ID), extBottom, font, 1.5, (255, 49, 12), 2)

		return img_arr
		


	def _zoomedBI_disp(self, seg_rect):
		## 1. Paste seg_rect to zoomed_BI (similar to _update_lineSeg_BI)
		common_bbox = self.zoomed_collect.common_bbox
		# create image block to paste on zoomed_BI
		boundary = [common_bbox[0][1], common_bbox[1][1], common_bbox[0][0], common_bbox[1][0]]  # [minR, maxR, minC, maxC]
		block = self.BiBlock()
		block.boundary = boundary
		block.BI = seg_rect

		if self.zoomed_BI is None:
			self.zoomed_BI = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
			self.zoomed_BI[common_bbox[0][1]:common_bbox[1][1], common_bbox[0][0]:common_bbox[1][0]] = seg_rect
		else:
			self.zoomed_BI = self._merge_into_BI(self.zoomed_BI, block)

		# 2. Display zoomed_BI (similar to _update_disp_line2 and _get_lineSeg_Disp)
		ori_img = self.img_arr  # numpy array

		if not (ori_img is None):
			# 1) draw BI on image
			bi_img = Image.fromarray(self.zoomed_BI)
			img = Image.fromarray(ori_img)
			draw = ImageDraw.Draw(img, mode='RGBA')
			draw.bitmap((0, 0), bi_img, fill=(0, 255, 0, 150))
			disp = np.array(img)
			self.img_arr = np.array(
				ori_img * (1 - self.alpha) + self.alpha * disp, dtype=np.uint8)

		# image = Image.fromarray(self.img_arr[:,:,[2,1,0]])
		image = Image.fromarray(self.img_arr)
		image.show()
		# self.canvas.setPixmap(self.img_arr)


	# def draw_animated_dot(self, center, r, colorFill, colorLine):
	# 	"""
	# 	center -- (y, x)
	# 	Draw animated dot on self.mousePaint_layer
	# 	"""
	# 	# clear all former dots
	# 	if self.lineSeg_Disp is None:
	# 		self._get_lineSeg_Disp()
	# 	self.img_arr = np.array(
	# 			self.ref_pic.img_arr * (1 - self.alpha) + self.alpha * self.lineSeg_Disp, dtype=np.uint8)
	# 	# draw dot on current position
	# 	self.simple_draw_circle(center, r, colorFill, colorLine)


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
		self.label_type = label_type
		self.load_label_info(label_type)
		# self.load_line_info()
		self.seg_method = seg_method
		self.init_seg()  # dont use label_type; init seg_arr with ignored label
		self.update_disp()  # dont use label_type
		self.update()  # dont use label_type
	# def update(self):
	#     super(SegPic, self).update()


	def _collect_info(self, cursorP):
		# 1. collect center
		y, x = cursorP
		# 2. collect left, right Edge Point
		leftEx = self.edge_x[0]
		rightEx = self.edge_x[1]
		# 3. collect sub_bbox
		TLx = x - self.rectW/2
		TLy = y - self.rectH/2
		BRx = x + self.rectW/2
		BRy = y + self.rectH/2

		TLx = 0 if TLx < 0 else TLx
		TLy = 0 if TLy < 0 else TLy
		BRx = self.w-1 if BRx >= self.w else BRx
		BRy = self.h-1 if BRy >= self.h else BRy

		# if the image is ZOOMED, collect all info into self.zoomed_collect else self.collect
		if self.Zoomed == True:
			if self.zoomed_collect is None:
				self.zoomed_collect = self.Collected(self.w, self.h)
			collect = self.zoomed_collect
		else:
			if self.collect is None:
				self.collect = self.Collected(self.w, self.h)
			collect = self.collect
			
		collect.center.append(self.cursorP)
		# 4. collect common_bbox
		[TL, BR] = collect.common_bbox
		TL[1] = min(TL[1], self.cursorP[0] - self.rectW/2)
		# print "cursorP: ", cursorP
		# print "(rectW/2, rectH/2): ", (self.rectW/2, self.rectH/2)
		TL[0] = min(TL[0], self.cursorP[1] - self.rectW/2)
		BR[1] = max(BR[1], self.cursorP[0] + self.rectH/2)
		BR[0] = max(BR[0], self.cursorP[1] + self.rectH/2)
		collect.common_bbox = [TL, BR]
		collect.leftE.append( (y, leftEx) )
		collect.rightE.append( (y, rightEx) )			
		collect.sub_bbox.append([(TLx, TLy), (BRx, BRy)])


	def _clear_collect_info(self, collect):
		collect.center = []
		collect.leftE = []
		collect.rightE = []
		collect.common_bbox = [[self.w-1, self.h-1], [0, 0]]
		collect.sub_bbox = []


	# def _point_small2large_uncrop(self, pointS_list):
	# 	out = []
	# 	for p in pointS_list:
	# 		# print p
	# 		pointL = [int(p[0] * self.zRate), int(p[1] * self.zRate)]
	# 		out.append(pointL)
	# 	return out

	
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
		# pointL -- (x, y)
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


	def load_line_info(self):
		mydir = os.path.dirname(os.path.realpath(__file__))
		info_path = mydir + "/data/line_Info.json"

		print "Loading: ", info_path
		if os.path.exists(info_path):
			d = read_json(info_path)
			if ('line_types' in d.keys()):
				t_list = d['line_types']
				for t in t_list:
					self.line_types.update(t)
				print "line_types-- ", self.line_types

			if ('normal' in d.keys()):
				self.normal = d['normal']
				print "normal_line-- ", self.normal

			if ('highlight' in d.keys()):
				self.highlight = d['highlight']
				print "highlight-- ", self.highlight

			if ('cutline' in d.keys()):
				self.cutline = d['cutline']
				print "cutline-- ", self.cutline

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


	# def undo_seg(self):
	# 	"""
	# 	pop from biblock_stk; set the all 1s in bi_rect as 0s
	# 	!!! This method also updated lineSeg_BI
	# 	"""
	# 	if self.cur_line_type2 in self.lineSeg_BI_dict.keys():
	# 		if len(self.lineSeg_BI_dict[self.cur_line_type2][0]) > 0:
	# 			biblock = self.lineSeg_BI_dict[self.cur_line_type2][0].pop(-1)
	# 			final_BI = self.lineSeg_BI_dict[self.cur_line_type2][1]
	# 			rect = biblock.boundary
	# 			bi = biblock.BI
	# 			# Image.fromarray(bi).show()
	# 			tmp = np.zeros(final_BI.shape, np.uint8)
	# 			tmp[rect[0] : rect[1], rect[2] : rect[3]] = bi

	# 			final_BI[tmp != 0] = 0

	# 			## update lineSegDisp_dict (must after lineSeg_BI_dict updated!!!)
	# 			self.lineSegDisp_dict[self.cur_line_type2] = self._get_lineSeg_Disp(self.cur_line_type2)

	# 			self.update_disp()
	# 			self.update()

	

	def pop_collect_poly_points(self):
		## similar to mouseMoveEvent -- mouseRightClick --> pop point from self.collect_poly_points
		if self.Zoomed == True:
			collect_poly_points = self.zoomed_collect_points
		else:
			collect_poly_points = self.collect_poly_points

		line_type = self.cur_line_type
		if (self.parent().edit_method == "line2" and self.parent().line2_mode == "polygonPlus"):
			line_type = "fnc_polygonPlus"
		elif (self.parent().edit_method == "line2" and self.parent().line2_mode == "polygonMinus"):
			line_type = "fnc_polygonMinus"
		elif self.parent().line2_mode == "polygonOtsu":
			line_type = "fnc_polygonOtsu"
		elif self.parent().line2_mode == "select":
			line_type = "fnc_select"

		if len(collect_poly_points) > 0:
			collect_poly_points.pop()
			k = len(collect_poly_points)
			# print "len of collect_pp = ", k
			self.update_disp()
			if k > 1:
				self.draw_polyline(
					collect_poly_points, [], line_type, False)
			self.update()


	def event(self, event):
		if self.parent().mode != "edit":
			return QtGui.QLabel.event(self, event)

		et = event.type()
		# print self.parent()
		edit_method = self.parent().edit_method
		line2_mode = self.parent().line2_mode if edit_method == "line2" else None
		# <G + leftMouseBtn> select a line
		# True if in [select line mode]
		select_line = self.parent().select_line
		# <C + leftMouseBtn> setting cutline's end point
		cutline_mode = self.parent().cutline_mode  # True if in [cut line mode]
		# <B + leftMouseBtn> start drawing bbox (rubberBand)
		bbox_mode = self.parent().bbox_mode  # True if in [bbox mode]
		# Use <space> to switch between positive_point/negative_point
		put_positive = self.parent().posi_point_mode

		line_type = self.cur_line_type2
		if edit_method == "line":
			line_type = self.cur_line_type
		if (edit_method == "line2" and self.parent().line2_mode == "polygonPlus"):
			line_type = "fnc_polygonPlus"
		elif (edit_method == "line2" and self.parent().line2_mode == "polygonMinus"):
			line_type = "fnc_polygonMinus"
		elif line2_mode == "polygonOtsu":
			line_type = "fnc_polygonOtsu"
		elif line2_mode == "select":
			line_type = "fnc_select"

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
					# !!! no need to self.update_disp() and self.update() because already contained in reset_current_superpixel
				elif edit_method == 'polygon' or edit_method == 'line':
					if collect_poly_points:
						last_breakp = ()   # empty tuple
						if (len(self.break_i) - self.i_mark > 0):
							last_breakp = collect_poly_points[self.break_i[-1] - self.i_mark]
						last_point = collect_poly_points.pop()
						# if the popped point is a break point
						if last_breakp == last_point:
							print "pop break point !!!"
							self.break_i.pop()

						# print "POP POINT!"
						self.update_disp()
						# if self.Zoomed == False:
						# 	self.update_disp()
						# else:
						# 	self.img_arr = self.ref_pic.img_arr.copy()

						k = len(collect_poly_points)
						# print "len of collect_pp = ", k
						if k > 1:
							self.draw_polyline(
								collect_poly_points, self.break_i, self.cur_line_type, False)

							# todraw = []
							# for i in range(k-1):
							# 	todraw.append((self.collect_poly_points[i],
							# 				   self.collect_poly_points[i+1]))
							# self.draw_lines(todraw, w, color)
						self.update()

				elif edit_method == 'line2' and self.ImgLoaded == True:
					##### TO DO...
					pass

					# # 1) If we've just created BBox or points and not yet Run
					# # Just clear BBox and points
					# if (self.cur_lineSeg_BI is None and len(self.bbox) == 2):
					# 	print "Clear BBox and points..."
					# 	self.pos_px = set()
					# 	self.neg_px = set()
					# 	self.bbox = []
					# 	self.update_disp_line2()
					# 	self.update()
					# 	return QtGui.QLabel.event(self, event)

					# # 2) Undo last Run_grabcut
					# # self.From_stk = True    # reload segline_disp
					# self.pos_px = set()
					# self.neg_px = set()
					# self.bbox = []
					# if (self.cur_line_type2 in self.lineSeg_BI_dict.keys()):
					# 	print "Rewind lineSeg_BI..."
					# 	# Rewind
					# 	# 1) pop from BiBlock_stk
					# 	BiBlock_stk = self.lineSeg_BI_dict[self.cur_line_type2][0]
					# 	print "Right Click: len(BiBlock_stk) -- ", len(BiBlock_stk) 
					# 	if (len(BiBlock_stk) > 1):
					# 		BiBlock_stk.pop()
					# 		BI = BiBlock_stk[-1].BI
					# 		self.cur_lineSeg_BI = np.zeros(self.ref_pic.img_arr.shape)
					# 		self.cur_lineSeg_BI[ BI != 0] = 255

					# 	elif (len(BiBlock_stk) == 1):
					# 		BiBlock_stk.pop()
					# 		self.cur_lineSeg_BI = None
					# 	# 2) reforming final_BI from lineSeg_BI_stk
					# 	final_BI = self.lineSeg_BI_dict[self.cur_line_type2][1]
					# 	final_BI[:] = 0
					# 	for block in BiBlock_stk:
					# 		final_BI = self.merge_into_BI(final_BI, block)
					# self.update_disp_line2()
					# self.update()
					# # self.From_stk = False


			# LEFT Click
			elif (event.buttons() == QtCore.Qt.LeftButton):
				# if edit_method == 'polygon' or edit_method == 'line' or (edit_method == "line2" and (self.parent().line2_mode == "polygonPlus" or self.parent().line2_mode == "polygonMinus")):
				# if edit_method == 'polygon' or edit_method == 'line' or (line2_mode == "polygonPlus" or line2_mode == "polygonMinus" or line2_mode == "polygonOtsu" or line2_mode == "select"):
				if self.MannualMode == True:
						self.cursorP[0] = event.pos().y()
						self.cursorP[1] = event.pos().x()
				# [Mode: Select Line]
				if (select_line == True):
					if edit_method == "line":
						# 1) clear collect_poly_points if previous line drawing
						# action isn't finished
						collect_poly_points = []
						self.break_idx = []
						self.update_disp()
						self.update()
						# self.clear_boundary_info()  # clear xmin, xmax, ....
						# 2) pick the nearest line by clicking on img
						pick_p = [event.pos().x(), event.pos().y()]
						# set self.selected_line with the nearest line
						self.select_a_line(pick_p)
						self.selected_line_disp(True)

						print "pick point: ", pick_p
						# print "--selected_line: ", self.selected_line.points
						# print "--selected_line center: ",
						# self.selected_line.center
						self.parent().exit_select_line_mode = True
						return QtGui.QLabel.event(self, event)

					# elif edit_method == "line2":
					# 	# self.cursorP (y, x)
					# 	mouseXY = [event.pos().x(), event.pos().y()]
					# 	## list of all indices of selected blob [[[x, y], [x, y], [x, y], ...], [], [], ...]
					# 	self.selected_lineSeg_indices = []  # clear
					# 	self.selected_lineSeg_rect = []  # clear
					# 	# rect: [min_r, max_r, min_c, max_c]
					# 	selectCoords, rect = self.selectBlob(mouseXY)
					# 	if len(selectCoords) != 0:
					# 		self.selected_lineSeg_indices.append(selectCoords)
					# 		self.selected_lineSeg_rect.append(rect)

					# 	self.update_disp()
					# 	self.update()

					# from [select_line mode] --> [draw_line mode]: line
					# highlight display off
				if (self.parent().exit_select_line_mode):
					print ">>selected off<<"
					self.update_disp()
					self.selected_line_disp(False)
					self.selected_line = None  # clear selected line
					self.parent().exit_select_line_mode = False

				# [(p1x, p1y), (p2x, p2y), ...]
				collect_poly_points.append(
					[self.cursorP[1], self.cursorP[0]])

				if len(collect_poly_points) > 1:
					p1 = collect_poly_points[-2]
					p2 = collect_poly_points[-1]

					print "draw line --> p1, p2: ({}, {})".format(p1, p2)
					# [Mode: setting cutline]
					# if (edit_method == 'line' and cutline_mode and self.allow_cutline):
					# 	# draw gray semi-trasparent line for break point
					# 	print "cut line: ", (p1, p2)
					# 	self.draw_polyline(
					# 		[p1, p2], [0], self.cur_line_type, False)
					# 	self.break_i.append(
					# 		len(collect_poly_points) - 2 + self.i_mark)
					# 	self.allow_cutline = False
					# else:
					# 	self.draw_polyline(
					# 		[p1, p2], [], self.cur_line_type, False)
					# 	self.allow_cutline = True
					if (edit_method == 'line' and cutline_mode):
						# draw gray semi-trasparent line for break point
						print "cut line: ", (p1, p2)
						self.draw_polyline(
							[p1, p2], [0], line_type, False)
						self.break_i.append(
							len(collect_poly_points) - 2 + self.i_mark)
					else:
						print "################## ", line2_mode
						self.draw_polyline(
							[p1, p2], [], line_type, False)

					self.update()

				if edit_method == 'line2' and self.parent().line2_mode == "smartGC" and self.ImgLoaded == True:
					pos = event.pos()
					x = pos.x()
					y = pos.y()
					
					if x < 0:
						x = 0
					if x >= self.w:
						x = self.w - 1
					if y < 0:
						y = 0
					if y >= self.h:
						y = self.h 
					# Get the mouse cursor position
					p = y, x
			
					if event.buttons() == QtCore.Qt.LeftButton:
						if self.MannualMode == True:
							self.cursorP = [p[0], p[1]]
						print "Collecting Point: {}".format(self.cursorP)
			
						self._collect_info(self.cursorP)

					# if bbox_mode == True:
					# 	## Start Drawing bbox
					# 	# 1) clear posive, negative points, curlineSeg_BI
					# 	self.pos_px = set()
					# 	self.neg_px = set()
					# 	self.cur_lineSeg_BI = None
					# 	# 2) clear current bbox on img
					# 	if len(self.bbox) != 0:
					# 		self.bbox = []
					# 		self.update_disp_line2()
					# 		self.update()
					# 	self.bbox.append((event.pos().x(), event.pos().y())) # Top Left of BBox
					# 	# print "bbox TL: ", self.bbox[-1]

					# else:
					# 	## Left click in BBox to put positive points (foreground points)
					# 	if put_positive == True:
					# 		mouse_pos = (event.pos().x(), event.pos().y())
					# 		self.add_point_4gc(mouse_pos, self.posi_dot_size, True)
					# 	## switch [space] Left click in BBox to put negative points (foreground points)
					# 	else:
					# 		mouse_pos = (event.pos().x(), event.pos().y())
					# 		self.add_point_4gc(mouse_pos, self.neg_dot_size, False)

		# elif (et == QtCore.QEvent.MouseButtonRelease):
		# 	if (event.button() == QtCore.Qt.LeftButton):
		# 		if edit_method == 'line2' and bbox_mode == True:
		# 			## Finish Drawing BBox
		# 			self.bbox.append((event.pos().x(), event.pos().y())) # Bottom Right of BBox
		# 			self.bbox = self.rect_intersect(self.bbox, [(0, 0), (int(self.img_arr.shape[1]), int(self.img_arr.shape[0]))])
		# 			# generate negative pixels automatically by setting 2*2 corners of bbox;
		# 			# bbox is invalid if we fail to generate neg_px automatically
		# 			if (self.auto_neg_px() == False):
		# 				self.bbox = []
		# 				return QtGui.QLabel.event(self, event)
		# 			self.draw_bbox()
		# 			self.update()                    


		return QtGui.QLabel.event(self, event)


	def mouseMoveEvent(self, event):
		if event.buttons() == QtCore.Qt.NoButton and self.parent().edit_method == "line2" or self.parent().edit_method == "line":
			## For cursor snap to line middle
			pos = event.pos()
			x = pos.x()
			y = pos.y()
			# x, y = pos.x()-self.img_arr.shape[0], pos.y()-self.img_arr.shape[1]
			# print "(x, y) = ", (x, y)

			if self.ImgLoaded == False or x < 0 or x >= self.w or y < 0 or y >= self.h:
				# print "!!! mouse cursor out of img area {}".format((x, y))
				pass
			else:
				p = y, x
				# look up p in self.hLookup
				i = 0
				while (i < len(self.hLookup2) and self.hLookup2[i] > p[0]):
					i += 1
				# (self.hLookup[i] is the smallest element >= y)
				# H, W of bbox
				## For Linear Shrink
				self.rectH = int(self.h0 - self.hDelta*(i-1))
				self.rectW = int(self.h0 - self.wDelta*(i-1))

				## For k^2 shrink
				# self.rectH = int(self.h0 * self.hRate**(i-1))
				# self.rectW = int(self.w0 * self.wRate**(i-1))

				if self.rectW < 10:
					self.rectW = 10
				if self.rectH < 10:
					self.rectH = 10

				# print p

				# ### Cursor Snap
				# # Get mid point of current row by doing canny edge detecting within local bbox
				# pMid = self._get_snap_point(p, self.rectH, self.rectW)
				# if pMid != p:
				# 	color = (255, 80, 0, 160)	# orange
				# 	# print "corrected cursor: ", pMid
				# else:
				# 	color = (94, 112, 142, 160)  # gray blue
				# 	# print "cursor: ", pMid
				

				# # Draw correct point pMid on Image
				# self.cursorP = [pMid[0], pMid[1]]
				# # print self.cursorP
				# if self.MannualMode == True:
				# 	self.cursorP = [y, x]
				# ### 

				### Not Using Cursor Snap
				pMid = self._get_snap_point(p, self.rectH, self.rectW)  ## used when calculate bbox in smartGC
				self.cursorP = [y, x]
				if self.parent().edit_method == "line2" and self.parent().line2_mode == "smartGC":
					color = (0, 161, 225, 220)
				elif self.parent().edit_method == "line2" and self.parent().line2_mode == "polygonPlus":
					color = (29, 178, 0, 220)
				elif self.parent().edit_method == "line2" and self.parent().line2_mode == "polygonMinus":
					color = (178, 0, 0, 220)
				else:
					color = (93, 14, 239, 220)
				###


				# print "<cursor position>: ", self.cursorP
				# self._draw_dots([self.cursorP], color)
				self.MannualMode == False

				self.visualP = [y, x]

		elif event.buttons() == QtCore.Qt.LeftButton:
			pt = event.pos()
			edit_method = self.parent().edit_method
			select_line = self.parent().select_line
			if edit_method == 'segmentation':
				self.collect_points.append((np.int(pt.x()), np.int(pt.y())))

			else:
				if(edit_method == 'polygon' or edit_method == 'line'):
				# if (edit_method == 'line' and select_line == True):
				#   pick_p = [np.int(pt.x()), np.int(pt.y())]
				#   print "pick point: ", pick_p
				#   return

					self.collect_points.append((np.int(pt.x()), np.int(pt.y())))

		elif event.buttons() == QtCore.Qt.RightButton:
			pass


	def mouseDoubleClickEvent(self, event):
		if self.ImgLoaded == False or self.parent().mode != 'edit':
			return

		if event.buttons() == QtCore.Qt.LeftButton:
			if self.parent().edit_method == "line2" and self.parent().line2_mode == "smartGC":
				# if zoomed image, using zoomed_collect else collect
				if self.Zoomed == True:
					collect = self.zoomed_collect
				else:
					collect = self.collect
	
				if not (collect is None and len(collect.center) <= 1):
					# print "len of collected centers", len(self.collect.center)
					print "collected centers: ", collect.center
					## 1. validate common_bbox
					self._anti_outofbound(collect.common_bbox)
					if (self._check_bbox_too_large(collect.common_bbox) == True):
						print "!!! Selected area too big ( > 1/5 of original size) !!!"
						print "Please reselect a smaller area !!!"
						self._clear_collect_info(collect)
						return
					## 2. Run GrabCutz
					self.grabcut = MyGrabCut(collect, self.ref_pic.img_arr)
					# Also modify Common_bbox (because it has been shrinked during grabcut.run)
					(seg_rect, collect.common_bbox) = self.grabcut.run_dilate()	# 0 or 255; numpy arr of img seg on shrinked_common_bbox area
					seg_rect[seg_rect != 0] = self.cur_line_label
					# seg_rect_img = Image.fromarray(seg_rect)
					# seg_rect_img.show()
					## rect: [row_upper, row_lower, col_left, col_right]
					rect = [collect.common_bbox[0][1], collect.common_bbox[1][1], collect.common_bbox[0][0], collect.common_bbox[1][0]]
					if self.Zoomed == True:
						# self.ZoomedModified = True	# final_BI has been modified during zoom in
						## resize(zoom out) seg_rect and rect, and update on original sized lineSeg_BI
						rh = (rect[1] - rect[0]) / self.zRate
						rw = (rect[3] - rect[2]) / self.zRate
						ry = (self.zoom_pos[1] + rect[0]) / self.zRate
						rx = (self.zoom_pos[0] + rect[2]) / self.zRate

						rect = [ry, ry + rh, rx, rx + rw]
						seg_rect_img = Image.fromarray(seg_rect, 'L')
						small_seg_rect_img = seg_rect_img.resize((rw, rh), Image.NEAREST)
						seg_rect = np.array(small_seg_rect_img)
						# print ">>>> line2_mode: ", self.parent().line2_mode
						# self._zoomedBI_disp(seg_rect)

					print "rect>> ", rect
					self.update_lineSeg_BI(seg_rect, rect)
					## 4. Update Disp
					self.update_disp()

				## 4. clear all collected list
				self._clear_collect_info(collect)

			elif len(self.collect_poly_points) >= 2 or len(self.zoomed_collect_points) > 0:
				if self.parent().edit_method == "line":
					self.update_line_value()

				# elif self.parent().edit_method == "polygon" or (self.parent().edit_method == "line2" and (self.parent().line2_mode == "polygonPlus" or self.parent().line2_mode == "polygonMinus")):
				elif self.parent().edit_method == "polygon" or (self.parent().edit_method == "line2" and (self.parent().line2_mode[:7] == "polygon" or self.parent().line2_mode == "select")):
					# self.parent().sp.setValue(self.cur_line_ID)
					# self.parent().sp.setGeometry(event_x, event_y, 10, 3)
					# self.parent().addWidget(self.parent().sp)
					# print ">>>>>> update polygon!!!"
					self.update_polygon_value()
					self.update_disp()

			self.update()


	def show_conflict_msgBox(self, ID):
		illegal_msg = QtGui.QMessageBox()
		illegal_msg.setIcon(QtGui.QMessageBox.Critical)
		illegal_msg.setText("lineID-lineType conflict")
		info_str = u'同一ID的线必须具有同样的种类(line_type)，检查 ID = {} 的线\n'.format(ID)
		info_str2 = u"All line with same ID should have same type!!! Check line with ID {}\n.".format(ID) 
		illegal_msg.setInformativeText(info_str + info_str2)
		illegal_msg.setWindowTitle("Conflict: line ID vs line Type")
		illegal_msg.setStandardButtons(QtGui.QMessageBox.Ok)
		illegal_msg.exec_()


	def disp_finalID_on(self):
		print "displaying self.final_ID ON"
		## unable edit
		self.parent().mode = "view"
		self.img_arr_tmp = self.img_arr.copy()

		# img_arr = int8_to_uint8(self.final_ID, self.int8_to_uint8_OFFSET)
		img_arr = np.zeros(self.final_ID.shape, np.uint8)
		img_arr[self.final_ID == self.cur_line_ID] = 255
		img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)

		self.img_arr = img_arr
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


		if self.cur_line_type2 == "all":
			## display color BI img
			segBI = np.zeros(img_arr.shape, np.uint8)
			segBI_img = Image.fromarray(segBI)
			draw = ImageDraw.Draw(segBI_img, mode='RGB')

			bi_t = np.zeros(self.final_BI.shape, np.uint8)
			for t in self.line_types.keys():
				if "label" not in self.line_types[t].keys():
					continue
				label = self.line_types[t]["label"]
				if label not in self.final_BI:
					continue
				color_fill = self.line_types[t]["color_fill"][0]
				bi_t[:] = 0
				bi_t[self.final_BI == label] = 255 
				
				bi_t_img = Image.fromarray(bi_t)
				draw.bitmap((0, 0), bi_t_img, fill=tuple(color_fill))
				segBI = np.array(segBI_img)
		else:
			## display binary img
			segBI = np.zeros(img_arr.shape[:2], np.uint8)
			segBI[self.final_BI == self.cur_line_label] = 255
			segBI = cv2.cvtColor(segBI, cv2.COLOR_GRAY2RGB)


		if self.Zoomed == True:
			large_segBI = Image.fromarray(segBI).resize((self.w * self.zRate, self.h * self.zRate), Image.NEAREST)
			cropped_segBI = large_segBI.crop(tuple(self.zoom_pos))
			segBI = np.array(cropped_segBI)

		self.img_arr = segBI
		self.update()


	# def disp_segBI_on(self):
	# 	print "displaying segBI ON"
	# 	## unable edit
	# 	self.parent().mode = "view"
	# 	self.img_arr_tmp = self.img_arr.copy()
	# 	img_arr = self.ori_img.copy()

	# 	if self.cur_line_type2 != "all" and self.cur_line_type2 not in self.lineSeg_BI_dict.keys():
	# 		segBI = np.zeros(img_arr.shape[:2], np.uint8)
	# 	else:
	# 		if self.cur_line_type2 != "all":
	# 			final_BI = self.lineSeg_BI_dict[self.cur_line_type2][1]
	# 			segBI = final_BI.copy()
	# 			segBI[segBI != 0] = 255
	# 		else:
	# 			grayDelta = 255 / (len(self.line_types.keys())-1)
	# 			segBI = np.zeros(img_arr.shape[:2], np.uint8)
	# 			for t in self.line_types.keys():
	# 				if t not in self.lineSeg_BI_dict.keys():
	# 					continue
	# 				lineSeg_label = self.line_types[t]["label"]
	# 				final_BI = self.lineSeg_BI_dict[t][1]
	# 				segBI[final_BI != 0] = lineSeg_label * grayDelta

	# 	if self.Zoomed == True:
	# 		large_segBI = Image.fromarray(segBI).resize((self.w * self.zRate, self.h * self.zRate), Image.NEAREST)
	# 		cropped_segBI = large_segBI.crop(tuple(self.zoom_pos))
	# 		segBI = np.array(cropped_segBI)

	# 	self.img_arr = cv2.cvtColor(segBI,cv2.COLOR_GRAY2RGB)
	# 	self.update()



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
		# if self.img_arr is None:
		# 	self.img_arr = self.ref_pic.img_arr

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
		if self.parent().edit_method == "line":
			line_type = self.cur_line_type

		elif self.parent().edit_method == "line2":
			# copy finalBI to zoomed_BI
			### Assuming ONLY 1 line_type (cur_line_type2 set to "default")
			# if (self.cur_line_type2 in self.lineSeg_BI_dict.keys()):
			# 	self.zoomed_BI = self.lineSeg_BI_dict[self.cur_line_type2][1].copy()
			# 	img_BI = Image.fromarray(self.zoomed_BI, 'L')
			# 	large_BI = img_BI.resize((W, H), Image.NEAREST)
			# 	self.zoomed_BI_img = large_BI.crop(tuple(self.zoom_pos))
			# 	self.zoomed_BI = np.array(self.zoomed_BI_img)

			if self.parent().line2_mode == "smartGC":
				line_type = self.cur_line_type2
			else:
				line_type = "fnc_" + self.parent().line2_mode

		## 1. for all lines in line_set, draw their in-bound part
		# ## update lineSegDisp_dict
		# self.lineSegDisp_dict[self.cur_line_type2] = self._get_lineSeg_Disp(self.cur_line_type2)
		self.update_disp()


		## 2. draw polyline in self.collect_poly_points
		if len(self.collect_poly_points) > 1:
			large_points = self._point_small2large(self.collect_poly_points)  # project points on to large img
			# large_points = self._large_point_inbound(self.collect_poly_points, self.break_i)  # transform all points in bound
			self.draw_polyline(
				large_points, self.break_i, line_type, False)			


		if len(self.zoomed_collect_points) == 0 and len(self.collect_poly_points) != 0:
			point_large = self._point_small2large([self.collect_poly_points[-1]])[0]
			self.zoomed_collect_points.append(point_large)
			self.i_mark = len(self.collect_poly_points)-1
			# print ">>>>>>>>> zoomed_collect_points: {}".format(self.zoomed_collect_points)

		self.update()   # display img_arr


		# if self.parent().edit_method == "line2":
		# 	img_now = Image.fromarray(self.img_arr)
		# 	large_img_now = img_now.resize((W, H), Image.BILINEAR)
		# 	img_now_cropped = large_img_now.crop(tuple(self.zoom_pos))
		# 	self.img_arr = np.array(img_now_cropped)
		# 	self.update()   # display img_arr



	def zoom_out(self):
		if self.Zoomed == False:
			return

		self.Zoomed = False

		# 1. Set display img (cv2_image) as original sized image
		self.ref_pic.img_arr = self.ori_img.copy()

		self.i_mark = 0

		# 2. if exists, push points in zoomed_collect into original collect
		########################## WRONG!!! will not work!!!! 
		########################## should force to run GrabCut OR force to clear collected points if we haven't double click
		if self.parent().edit_method == "line2":
			if not (self.zoomed_collect is None or len(self.zoomed_collect.center) == 0):
				self._clear_collect_info(self.zoomed_collect)
	
			# # 3. if modified BI during zooming. zoom out zoomed_BI and save it back to lineSeg_BI_dict
			# if self.ZoomedModified == True and self.zoomed_BI is not None:
			# 	img = Image.fromarray(self.zoomed_BI, 'L')
			# 	w, h = img.size
			# 	W = w / self.zRate
			# 	H = h / self.zRate
			# 	small_img = img.resize((W, H), Image.NEAREST)
			# 	# print ">>>>>>>> show small_img"
			# 	# small_img.show()
			# 	# rect, [upper, lower, left, right]
			# 	rect = [self.zoom_pos[1]/self.zRate, self.zoom_pos[1]/self.zRate + H, self.zoom_pos[0]/self.zRate, self.zoom_pos[0]/self.zRate + W]  # [minR, maxR, minC, maxC]
			# 	# print "rect = ", rect
			# 	# print "small_img.size = ", small_img.size
			# 	self.update_lineSeg_BI(small_img, rect)
			
			# self._update_disp_line2()
			self.zoomed_BI = None
			if self.parent().line2_mode == "smartGC":
				line_type = self.cur_line_type2
			else:
				line_type = "fnc_" + self.parent().line2_mode

		if self.parent().edit_method == "line":
			line_type = self.cur_line_type

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
				self.collect_poly_points, self.break_i, line_type, False)
		self.zoomed_collect_points = []



		# elif self.parent().edit_method == "line" or (self.parent().edit_method == "line2" and (self.parent().line2_mode == "polygonPlus" or self.parent().line2_mode == "polygonMinus")):
		# 	for i, point in enumerate(self.zoomed_collect_points):
		# 		if i == 0 and len(self.collect_poly_points) > 0:
		# 			print "skip 1st point in zoomed_collect_points"
		# 			continue
		# 		else:
		# 			pointS = self._point_large2small([point])
		# 			# print "pointS = ", pointS				
		# 			self.collect_poly_points.extend(pointS)

		# 	print "collected points: ", self.collect_poly_points
		# 	## 1. draw all lines in line_set
		# 	self.update_disp()
		# 	## 2. draw all lines in collect_poly_points
		# 	if len(self.collect_poly_points) > 1:
		# 		# print "draw collect_poly_points --> {}".format(self.collect_poly_points)
		# 		self.draw_polyline(
		# 			self.collect_poly_points, self.break_i, self.cur_line_type, False)
		# 	self.zoomed_collect_points = []

		self.update()
		self.ZoomedModified = False


	def _get_common_bbox(self, sub_bbox):
		common_bbox = [[self.w-1, self.h-1], [0, 0]]
		for bbox in sub_bbox:
			common_bbox[0][0] = min(common_bbox[0][0], bbox[0][0])
			common_bbox[0][1] = min(common_bbox[0][1], bbox[0][1])
			common_bbox[1][0] = max(common_bbox[1][0], bbox[1][0])
			common_bbox[1][1] = max(common_bbox[1][1], bbox[1][1])

		return common_bbox


	def _anti_outofbound(self, bbox):
		# bbox -- [[TLx, TLy], [BRx, BRy]]
		bbox[0][0] = 0 if bbox[0][0] < 0 else bbox[0][0]
		bbox[0][1] = 0 if bbox[0][1] < 0 else bbox[0][1]
		bbox[1][0] = self.w-1 if bbox[1][0] > self.w-1 else bbox[1][0]
		bbox[1][1] = self.h-1 if bbox[1][1] > self.h-1 else bbox[1][1]


	def _check_bbox_too_large(self, bbox):
		# bbox -- [(TLx, TLy), (BRx, BRy)]
		bbox_size = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
		if (bbox_size > self.img_arr.size/6):
			print "large bbox: ", bbox
			return True
		return False


	# def auto_neg_px(self):
	# 	"""
	# 		Automatically set 4 'L'-shape corners(3 px/corner) on self.bbox as neg_px
	# 		return True if we're able to set all 4 'L'-shapes
	# 		- bbox has to be at least containing 3 * 3 = 9 valid pixels
	# 	"""
	# 	if (self.bbox[1][0] - self.bbox[0][0] < 3 or self.bbox[1][1] - self.bbox[0][1] < 3):
	# 		return False

	# 	# # TL
	# 	# self.neg_px.add((self.bbox[0][0], self.bbox[0][1]))
	# 	# self.neg_px.add((self.bbox[0][0]+1, self.bbox[0][1]))
	# 	# self.neg_px.add((self.bbox[0][0], self.bbox[0][1]+1))
	# 	# # TR
	# 	# self.neg_px.add((self.bbox[1][0]-1, self.bbox[0][1]))
	# 	# self.neg_px.add((self.bbox[1][0]-2, self.bbox[0][1]))
	# 	# self.neg_px.add((self.bbox[1][0]-1, self.bbox[0][1]+1))
	# 	# # BL
	# 	# self.neg_px.add((self.bbox[0][0], self.bbox[1][1]-1))
	# 	# self.neg_px.add((self.bbox[0][0]+1, self.bbox[1][1]-2))
	# 	# self.neg_px.add((self.bbox[0][0]+1, self.bbox[1][1]-1))
	# 	# # BR
	# 	# self.neg_px.add((self.bbox[1][0]-1, self.bbox[1][1]-1))
	# 	# self.neg_px.add((self.bbox[1][0]-1, self.bbox[1][1]-2))
	# 	# self.neg_px.add((self.bbox[1][0]-2, self.bbox[1][1]-1))

	# 	return True

	# 	self.draw_points(list(self.neg_px), False)
	# 	self.update()

	# def add_point_4gc(self, p, r, Positive=True):
	# 	"""
	# 		Form a dot with size r * r around p;
	# 		All pixels within the square dot should be positive pixels if within self.BBox
	# 		- 4 L-shape corners pixels of BBox are reserved for auto_neg pixels;
	# 	"""
	# 	if len(self.bbox) != 2 or self.p_out_of_bbox(p) == True:
	# 		print "point ", p, " out of bbox!!"
	# 		return

	# 	dot = []   # [(TLx, TLy), (BRx, BRy)] -- all pixels within this rect is pos_px
	# 	## 1) r is odd
	# 	if (r % 2 != 0):
	# 		dot.append(((p[0] - r/2), (p[1] - r/2)))
	# 		dot.append(((p[0] + r/2), (p[1] + r/2)))
	# 	## 2) r is even:
	# 	else:
	# 		dot.append(((p[0] - r/2 + 1), (p[1] - r/2 + 1)))
	# 		dot.append(((p[0] + r/2), (p[1] + r/2)))
	# 	print "Add dot: ", dot, " Positive: ", Positive

	# 	for c in range(dot[0][0], dot[1][0]+1):
	# 		for r in range(dot[0][1], dot[1][1]+1):
	# 			# print "(c,r): ", (c, r)
	# 			# check within bbox boundary
	# 			if not self.p_out_of_bbox((c, r)):
	# 				# check not in corner (auto_neg pixels)
	# 				if Positive == True:
	# 					if (c, r) not in self.neg_px:
	# 						self.pos_px.add((c, r))
	# 						self.draw_points(list(self.pos_px), True)
	# 				else:
	# 					if (c, r) not in self.pos_px:
	# 						self.neg_px.add((c, r))
	# 						self.draw_points(list(self.neg_px), False)
	# 	self.update()

	# def p_out_of_bbox(self, p):
	# 	c = p[0]
	# 	r = p[1]
	# 	if (c >= self.bbox[0][0] and c < self.bbox[1][0] and r >= self.bbox[0][1] and r < self.bbox[1][1]):
	# 		return False
	# 	return True


	# def rect_intersect(self, rect1, rect2):
	# 	"""
	# 		rect: [TopLeft, BottomRight]; TopLeft -- (TL_col, TL_row)
	# 		return rect (TL < BR)
	# 	"""
	# 	#### TBD
	# 	cols = [rect1[0][0], rect1[1][0], rect2[0][0], rect2[1][0]]
	# 	rows = [rect1[0][1], rect1[1][1], rect2[0][1], rect2[1][1]]
	# 	cols.sort()
	# 	rows.sort()

	# 	return [(cols[1], rows[1]), (cols[2], rows[2])]

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


	# def draw_bbox(self, img_arr, rect_list):
	# 	# rect in rect_list: [min_r, max_r, min_c, max_c]
	# 	outline_color = (255, 0, 0)
	# 	if (img_arr is not None):
	# 		img = Image.fromarray(img_arr)
	# 		draw = ImageDraw.Draw(img, mode='RGB')
	# 		for rect in rect_list:
	# 			print ">>>>>>>> draw rect: ", rect
	# 			# [x0, y0, x1, y1]
	# 			draw.rectangle([rect[2], rect[0], rect[3], rect[1]], outline=outline_color)

	# 		img.show()
	# 		return np.array(img)

	# 	return img_arr


	def draw_bbox(self):
		print "BBox: ", self.bbox
		# print "img_arr Size: ", self.img_arr.shape
		outline_c = (0, 0, 255, 250)

		if (len(self.bbox) < 2 or self.bbox[0] == self.bbox[1]):
			print "No rect to draw!!!"
			return

		if (self.img_arr is not None):
			img = Image.fromarray(self.img_arr)
			draw = ImageDraw.Draw(img, mode='RGBA')
			# draw.rectangle([992, 494, 952, 501], outline=outline_c)
			draw.rectangle(self.bbox, outline=outline_c)

			self.img_arr = np.array(img)

	def select_a_line(self, pick_p):
		# pick_p: [px, py]; position picked on img by G + mouseLeftBtn_click
		# set 'self.selected_line' with the nearest LaneLine in self.line_set from the pick_position
		# self.selected_line = None if line_set is empty

		if (self.line_set is not None and len(self.line_set) != 0):
			# Method 1. Select by center distance -- Time: O(N) N is # of polylines
			# self.selected_line = list(self.line_set)[0]
			# dist = self.p2p_distance(self.selected_line.center, pick_p)

			# for line in self.line_set:
			# 	tmp_dist = self.line_point_distance(line, pick_p)
			# 	if (tmp_dist < dist):
			# 		self.selected_line = line
			# 		dist = tmp_dist

			# Method 2. Select by checking if pick_p near line -- Time: O(M) M
			# is # of all tinylines
			for line in self.line_set:
				if self.Zoomed == True:
					points = self._point_small2large(line.points)
				else:
					points = line.points
				# [([p1x, p1y], [p2x, p2y]), ([p2x, p2y], [p3x, p3y]), ...] line with 2 ends
				tinyline = []
				for i in range(len(points) - 1):
					tinyline.append((points[i], points[i + 1]))
					if (self.check_near_line(tinyline, pick_p)):
						self.selected_line = line
						break


	def selectBlob(self, mouseXY):
		# Detect blobs on binary tmp_BI (0, 255)
		tmp_BI = np.zeros(self.final_BI.shape, np.uint8)
		tmp_BI[self.final_BI == self.cur_line_label] = 255

		all_labels = measure.label(tmp_BI, background=0)
		props = measure.regionprops(all_labels, tmp_BI)

		coord_RC = []
		coord_XY = []
		rect = []  # [min_r, max_r, min_c, max_c]
		for prop in props:
			## bbox: tuple (min_r, min_c, max_r, max_c)
			min_r, min_c, max_r, max_c = prop.bbox
			if mouseXY[0] >= min_c and mouseXY[0] <= max_c and mouseXY[1] >= min_r and mouseXY[1] <= max_r:
				print "got one blob !!!!"
				coord_RC = prop.coords.tolist()
				rect = [min_r, max_r, min_c, max_c]
				break

		## [row, col] to [x, y]
		for RC in coord_RC:
			coord_XY.append([RC[1], RC[0]])

		# print coord_XY
		return coord_XY, rect


	def p2p_distance(self, p1, p2):
		# calculate Euclidean dist between 2 points
		return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**(1 / 2.0)

	def check_near_line(self, tinyline, p):
		"""
		Check if point p is on (near) sublines
		@tinyline: [([p1x, p1y], [p2x, p2y]), ([p2x, p2y], [p3x, p3y]), ...] line with 2 ends forming polyline
		@p: [px, py]

		Idea:
		point 'p' is near tinyline <a, b> if
		-epsilon < (distance(a, p) + distance(p, b) - distance(a, b)) < epsilon
		"""
		dist_pb = self.p2p_distance(p, tinyline[0][0])
		for a, b in tinyline:
			dist_ab = self.p2p_distance(a, b)
			dist_pa = dist_pb
			dist_pb = self.p2p_distance(p, b)
			delta = dist_pa + dist_pb - dist_ab
			if delta > -self.mouse_epsilon and delta < self.mouse_epsilon:
				return True
		return False

	def selected_line_disp(self, highlight):
		"""
				Update display for self.selected_line
		"""
		if self.selected_line is None:
			return

		# self.update_disp()

		if self.Zoomed == True:
			points = self._point_small2large(self.selected_line.points)
		else:
			points = self.selected_line.points 
		self.draw_polyline(
			points,
			self.selected_line.break_idx,
			self.selected_line.line_type,
			highlight)
		self.update()

		# if (not self.img_arr is None):
		# 	todraw = []
		# 	for i in range(len(self.selected_line.points)-1):
		# 		todraw.append((self.selected_line.points[i], self.selected_line.points[i+1]))
		# 		self.draw_lines(todraw, w, self.line_info[self.selected_line.line_type])
		# 	self.update()

	def draw_polyline(self, line_points, breaki, t, highlight):
		"""
		Draw polyline
		@line: [(p1x, p1y), (p2x, p2y), ....]
		@breaki: index of break points on line [4, 8]
		@t: line_type name string
		@highlight: True/False
		"""
		# print ">>>>>>>>>>> ", t
		for p in line_points:
			if self._point_outofbound(p) == True and self.Zoomed == True:
				line_points, breaki = self._large_point_inbound(line_points, breaki)
				break

		print "draw points: {}".format(line_points)

		if (self.img_arr is not None) and (len(line_points) > 0):
			img = Image.fromarray(self.img_arr)
			draw = ImageDraw.Draw(img, mode='RGBA')

			j = 0
			for i in range(len(line_points) - 1):
				# default color
				color = [(255, 0, 0, 240)]
				# default width
				wi = [3]

				if (self.parent().edit_method == "line" or self.parent().edit_method == "line2"):
					color = []
					wi = []
					if (highlight):
						for c in self.line_types[t]["color"]:
							color.append(
								tuple(c + [self.highlight["opacity"]]))
						wi = self.highlight["width"][:len(color)]
					else:
						for c in self.line_types[t]["color"]:
							color.append(tuple(c + [self.normal["opacity"]]))
						wi = self.normal["width"][:len(color)]
						if (self.parent().edit_method == "line2" and (self.parent().line2_mode == "polygonMinus" or self.parent().line2_mode == "polygonPlus" or self.parent().line2_mode == "polygonOtsu" or self.parent().line2_mode == "select")):
							# override opacity and width
							wi = self.line_types[t]["width"]

				# print "color: ", color
				p1 = line_points[i]
				p2 = line_points[i + 1]
				if (j < len(breaki) and i == breaki[j]):
					# p1 is a break point, draw gray semi-opacity line between
					# (p1, p2)
					draw.line([p1[0], p1[1], p2[0], p2[1]], fill=(
						200, 200, 200, 110), width=wi[0])
					#draw.line([p1[0], p1[1], p2[0], p2[1]], fill=tuple(self.line_types["cutline"]["color"]), width=wi[0])
					j += 1

				else:
					for i in range(len(color)):
						draw.line([p1[0], p1[1], p2[0], p2[1]],
								  fill=color[i], width=wi[i])

			#font = ImageFont.truetype("sans-serif.ttf", 16)
			text_pos = ((line_points[-1][0] + line_points[-2][0]) / 2,
						(line_points[-1][1] + line_points[-2][1]) / 2)
			font = ImageFont.truetype("C:\WINDOWS\Fonts\SIMYOU.TTF", 20)
			draw.text(
				text_pos, unicode(
					self.line_types[t]["text"]), (219, 47, 1), font=font)

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
		tmp = self.seg_disp.reshape((-1, 3))  # 3 cols: R, G, B
		# different color represents diff labels
		tmp[selected_index, :] = self.color_map[label]
		self.seg_disp = tmp.reshape(self.seg_disp.shape)

		self.img_arr = np.array(self.ref_pic.img_arr *
								(1 -
								 self.alpha) +
								self.alpha *
								self.seg_disp, dtype=np.uint8)
		self.img_arr = np.array(
			mark_boundaries(
				self.img_arr,
				self.seg_index) * 255,
			dtype=np.uint8)
		self.collect_points = []  # reset collect_points[]
		self.update()

	def update_line_value(self):
		"""
				evoke when double click mouse after drawing the line
				1. Create LaneLine obj
				2. Store current LaneLine obj in self.line_set
		"""
		print "Updating_LaneLine_Value..."

		if self.Zoomed == True:
			self.zoom_out()

		if (len(self.collect_poly_points) > 1):
			self.cur_line = self.LaneLine()
			self.cur_line.line_type = self.cur_line_type

			# if (len(self.break_i) > 0):
			# 	# post-processing points
			# 	# 1) if break_idx contains heading continous numbers; e.g. [0, 1, 2, 5, ...]
			# 	#	 remove all heading points
			# 	f = 1
			# 	del_cnt = 0
			# 	if (self.break_i[0] == 0):
			# 		while (f < len(self.break_i)
			# 			   and self.break_i[f] == self.break_i[f - 1] + 1):
			# 			# remove element at index self.break_i[f-1]
			# 			self.collect_poly_points.remove(
			# 				self.collect_poly_points[self.break_i[f - 1] - del_cnt])
			# 			f += 1
			# 			del_cnt += 1
			# 		self.collect_poly_points.remove(
			# 			self.collect_poly_points[self.break_i[f - 1] - del_cnt])
			# 		if (f < len(self.break_i)):
			# 			self.cur_line.break_idx.append(
			# 				self.break_i[f] - del_cnt)
			# 	else:
			# 		self.cur_line.break_idx.append(self.break_i[0])

			# 	# 2) remove all continous break points
			# 	while (f < len(self.break_i)):
			# 		if (self.break_i[f] == self.break_i[f - 1] + 1):
			# 			# remove element at index self.break_i[f]
			# 			self.collect_poly_points.remove(
			# 				self.collect_poly_points[self.break_i[f] - del_cnt])
			# 			del_cnt += 1
			# 		else:
			# 			self.cur_line.break_idx.append(
			# 				self.break_i[f] - del_cnt)
			# 		f += 1

			# 	# 3) for the last point
			# 	# 	 if last_point is a break_point: delete the point
			# 	if (len(self.cur_line.break_idx) > 0):
			# 		if (self.cur_line.break_idx[-1] ==
			# 				len(self.collect_poly_points) - 1):
			# 			self.collect_poly_points.pop()
			# 			self.cur_line.break_idx.pop()
			# 		# elif last_point's previous point is a break_point: delete
			# 		# the point
			# 		elif (self.cur_line.break_idx[-1] == len(self.collect_poly_points) - 2):
			# 			self.collect_poly_points.pop()
			# 			self.cur_line.break_idx.pop()

			# print "collected break points: ", self.break_i
			# print "final break points: ", self.cur_line.break_idx
			# self.cur_line.points[:] = self.collect_poly_points
			for p in self.collect_poly_points:
				self.cur_line.points.append(list(p))
			self.cur_line.break_idx[:] = self.break_i
			print "cur_line.points: ", self.cur_line.points
			print "cur_line.break_idx: ", self.cur_line.break_idx

			# self.cur_line.MBR[:] = [[self.xmin[-1], self.ymin[-1]], [self.xmax[-1], self.ymax[-1]]]  # UL_point, BR_point
			# self.cur_line.cal_center()

			self.line_set.add(self.cur_line)
			self.collect_poly_points = []
			self.break_i = []
			# self.clear_boundary_info()  # clear xmin, xmax, ....

			self.update_disp()
			self.update()


	# def grabcut_on_bbox(self):
	# 	if (len(self.bbox) < 2 or (len(self.pos_px) == 0 and len(self.neg_px) == 0)):
	# 		print "[GrabCut_on_BBox]: No BBox or no pixels!!!"
	# 		print "[GrabCut_on_BBox]: Load original image instead..."
	# 		return
	# 	print "pos_px: ", list(self.pos_px)
	# 	print "neg_px: ", list(self.neg_px)

	# 	## 1. Crop BBox area Img
	# 	crop_bound = (self.bbox[0][0], self.bbox[0][1], self.bbox[1][0], self.bbox[1][1])
	# 	ori_img = Image.fromarray(self.ref_pic.img_arr)
	# 	bbox_img = ori_img.crop(crop_bound)
	# 	bbox_img_array = np.array(bbox_img)
	# 	bbox_img_array = bbox_img_array[:, :, ::-1].copy()  # RGB to BGR
	# 	# Image.fromarray(bbox_img_array).show()

	# 	## 2. Generate mask for GrabCut from pos_px, neg_px
	# 	mask = np.zeros(self.ref_pic.img_arr.shape[:2],np.uint8)
	# 	mask[:] = 2
	# 	# 2.1) positive as 0
	# 	for p in list(self.pos_px):
	# 		mask[p[1]][p[0]] = 1
	# 	if (len(self.pos_px) == 0):
	# 		mask[self.bbox[0][1]][self.bbox[0][0]] = 1

	# 	# 2.2) negative as 1
	# 	for p in list(self.neg_px):
	# 		mask[p[1]][p[0]] = 0
	# 	if (len(self.neg_px) == 0):
	# 		mask[self.bbox[0][1]][self.bbox[0][0]] = 0

	# 	# 2.3) crop mask
	# 	mask = mask[self.bbox[0][1]:self.bbox[1][1], self.bbox[0][0]:self.bbox[1][0]]
	# 	print "mask shape: ", mask.shape

	# 	# ##### TEST: show mask
	# 	# mask_test = np.zeros(mask.shape, np.uint8)
	# 	# mask_test[ mask == 2] = 100
	# 	# mask_test[ mask == 0] = 0
	# 	# mask_test[ mask == 1] = 255
	# 	# mask_show = Image.fromarray(mask_test)
	# 	# # mask_show.show()
	# 	# #####

	# 	bgdModel = np.zeros((1,65),np.float64)
	# 	fgdModel = np.zeros((1,65),np.float64)

	# 	mask, bgdModel, fgdModel = cv2.grabCut(bbox_img_array, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
	# 	mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

	# 	# 3. Update cur_lineSeg_BI
	# 	self.update_lineSeg_BI(mask)

	# 	# 4. Switch from putting negative --> positive
	# 	self.parent().posi_point_mode = True 

	# 	# 5. Load Mask on Image self.bbox block
	# 	self.update_disp_line2()
	# 	self.update()

	# 	# ##### TEST: show mask | show bbox_img_out
	# 	# mask_test = np.zeros(mask.shape, np.uint8)
	# 	# mask_test[:] = 100
	# 	# mask_test[ mask == 0] = 0
	# 	# mask_test[ mask == 1] = 255
	# 	# mask_show = Image.fromarray(mask_test)
	# 	# # mask_show.show()

	# 	# bbox_img_array = bbox_img_array * mask[:,:,np.newaxis]
	# 	# # print "bbox_img_array.shape: ", bbox_img_array.shape 
	# 	# bbox_img_array = bbox_img_array[:,:,[2,1,0]].copy()   
	# 	# bbox_img_out = Image.fromarray(bbox_img_array)
	# 	# # print "bbox_img_out.size: ", bbox_img_out.size
	# 	# # bbox_img_out.show()
	# 	# #####


	def update_lineSeg_BI(self, seg_rect, rect, old_label, label, old_ID, ID):
		# boundary, rect: [left, right, upper, lower]
		# 1) update BiBlock_stk and final_BI of cur_line_type2
		block = self.BiBlock()
		block.boundary = rect
		block.BI = seg_rect
		## make sure block.BI is marked with cur_line_label not 255
		block.BI[block.BI != 0] = label

		self.final_BI = np.zeros(img_arr.shape[:2], np.uint8) if self.final_BI is None else self.final_BI
		## keep final_ID sync with final_BI
		if self.final_ID is None:
			self.final_ID = np.zeros(img_arr.shape[:2], np.int8)
			self.final_ID[:] = -self.int8_to_uint8_OFFSET   # background ID
		# old_ID = self.find_old_ID(rect)
		# old_label = self.find_old_label(rect)
		if (self.parent().line2_mode == "select" or self.parent().line2_mode == "polygonMinus") and (old_label != self.cur_line_label and old_label != 0):
			print ">>>>>old_label: {}".format(old_label)
			# print ">>>>>line_types.keys(): {}".format(self.line_types.keys())
			logging_str = "--You can only edit lineType <{}> when selected lineType <{}>.\n".format(self.line_types.keys()[old_label-1], self.line_types.keys()[old_label-1])
			self.parent().add_logging(logging_str)
			logging_strC = u'--要编辑种类(line_type)为 <{}> 的线，需要在右侧选择到相应的种类!F\n'.format(self.line_types.keys()[old_label-1])
			self.parent().add_logging(logging_strC)
			return
		# 1) merge into final_BI
		self.final_BI, self.final_ID = self._merge_into_BI(self.final_BI, self.final_ID, block, label, old_ID, ID)

		# print "@@@@@@@@@@@@@@@@@@>>>>>>>>> ", np.count_nonzero(self.final_BI)
		# final_BI = np.zeros(self.final_BI.shape, np.uint8)
		# final_BI[self.final_BI != 0] = 255
		# Image.fromarray(final_BI, 'L').show()

		# 2) append to BiBlock_stk
		if (np.count_nonzero(seg_rect) != 0) and (self.parent().line2_mode != "select"):
			if len(self.BiBlock_stk) == self.UndoCacheMaxSize:
				# remove from stack bottom
				self.BiBlock_stk.pop(0)
			self.BiBlock_stk.append(block)
			print "pushing into BiBlock_stk: ", len(self.BiBlock_stk)

		self._ID2label_dict_del(old_ID, old_label)
		if not self.parent().line2_mode == "polygonMinus":
			self._ID2label_dict_add(ID, label)

		# if (self.parent().line2_mode == "select"):
		# 	self._update_ID2label(old_ID, self.cur_line_label)
		# self._update_ID2label(ID, label)
		## also update self.lineSegDisp_dict (must after self.lineSeg_BI_dict[self.cur_line_type2] updated)
		# self.lineSegDisp_dict[self.cur_line_type2] = self._get_lineSeg_Disp(self.cur_line_type2)
		# Image.fromarray(self.lineSeg_BI_dict[self.cur_line_type2][1]).show()



	# def update_lineSeg_BI(self, seg_rect, rect):
	# 	# boundary, rect: [left, right, upper, lower]
	# 	# 1) update BiBlock_stk and final_BI of cur_line_type2 in lineSeg_BI_dict
	# 	block = self.BiBlock()
	# 	block.boundary = rect
	# 	block.BI = seg_rect

	# 	img_arr = self.ori_img.copy()  ## always update on original_sized image (not zoomed img)

	# 	if (self.cur_line_type2 in self.lineSeg_BI_dict.keys()):
	# 		# 1) append to BiBlock_stk (tuple[0])
	# 		if np.count_nonzero(seg_rect) != 0:
	# 			if len(self.lineSeg_BI_dict[self.cur_line_type2][0]) == self.UndoCacheMaxSize:
	# 				# remove from stack bottom
	# 				self.lineSeg_BI_dict[self.cur_line_type2][0].pop(0)
	# 			self.lineSeg_BI_dict[self.cur_line_type2][0].append(block)
	# 			print "pushing into BiBlock_stk: ", len(self.lineSeg_BI_dict[self.cur_line_type2][0])
	# 		# self.lineSeg_BI_dict[self.cur_line_type2][0].append(block)
	# 		# 2) merge into lineSeg_BI_final (tuple[1])
	# 		final_BI = self.lineSeg_BI_dict[self.cur_line_type2][1] 
	# 		final_BI = self._merge_into_BI(final_BI, block)
	# 		self.lineSeg_BI_dict[self.cur_line_type2][1] = final_BI
	# 		# Image.fromarray(self.lineSeg_BI_dict[self.cur_line_type2][1]).show()
	# 	else:
	# 		print "creating BiBlock_stk..."
	# 		BiBlock_stk = []
	# 		final_BI = np.zeros(img_arr.shape[:2], np.uint8)
	# 		# print "seg_rect.shape: ", seg_rect.shape
	# 		# print "rect = ", rect
	# 		# print ">>>>>>>>>>>>>> 2 rect: ", rect
	# 		final_BI[rect[0] : rect[1], rect[2] : rect[3]] = seg_rect
	# 		if np.count_nonzero(seg_rect) != 0:
	# 			BiBlock_stk.append(block)
	# 		self.lineSeg_BI_dict[self.cur_line_type2] = [BiBlock_stk, final_BI]

	# 	## also update self.lineSegDisp_dict (must after self.lineSeg_BI_dict[self.cur_line_type2] updated)
	# 	self.lineSegDisp_dict[self.cur_line_type2] = self._get_lineSeg_Disp(self.cur_line_type2)

	# 	# Image.fromarray(self.lineSeg_BI_dict[self.cur_line_type2][1]).show()


	def _merge_into_BI(self, final_BI, final_ID, block, label, oldID, ID):
		img_arr = self.ori_img.copy()  ## always update on original_sized image (not zoomed img)   
		boundary = block.boundary

		# print ">>>>>>>>>>> ", type(final_BI)		
		if self.parent().line2_mode == "smartGC" or self.parent().line2_mode == "polygonPlus" or self.parent().line2_mode == "polygonOtsu" or self.parent().line2_mode == "select":
			tmp = np.zeros(img_arr.shape[:2], np.uint8)
			tmp[boundary[0] : boundary[1], boundary[2] : boundary[3]] = block.BI
			# Image.fromarray(tmp, 'L').show()
			final_BI[tmp != 0] = label
			final_ID[tmp != 0] = ID
			# print "count nonzero -- tmp: {}".format(np.count_nonzero(tmp))
			# print "count nonzero -- final_BI: {}".format(np.count_nonzero(final_BI))
			# Image.fromarray(final_BI).show()
		elif self.parent().line2_mode == "polygonMinus":
			t_BI = np.zeros(img_arr.shape[:2], np.uint8)
			t_BI[final_BI == label] = label

			tmp = np.ones(img_arr.shape[:2], np.uint8)
			tmp[boundary[0] : boundary[1], boundary[2] : boundary[3]] = block.BI
			tmp[tmp != 0] = label
			t_BI[tmp == 0] = 0
			# t_BI[t_BI != 0] = 255
			# Image.fromarray(t_BI).show()

			# oldID = self.find_old_ID(block.boundary)

			final_ID[(t_BI == 0) & (final_ID == oldID) & (final_BI == label)] = -self.int8_to_uint8_OFFSET	 # background ID
			final_BI[(t_BI == 0) & (final_BI == label)] = 0		
			# final_ID[final_ID != 0] = 255
			# Image.fromarray(final_ID).show()

		return (final_BI, final_ID)


	# def find_old_ID(self, rect):
	# 	final_ID_rect = self.final_ID[rect[0] : rect[1], rect[2] : rect[3]]
	# 	unique, counts = np.unique(final_ID_rect, return_counts=True)
	# 	ID_cnt = dict(zip(unique, counts))
	# 	## sort by value; ascending order
	# 	sorted_dict = sorted(ID_cnt.items(), key=operator.itemgetter(1))

	# 	if sorted_dict[-1][0] == -self.int8_to_uint8_OFFSET:
	# 		if len(sorted_dict) > 1:
	# 			return sorted_dict[-2][0]
	# 		else:
	# 			return -self.int8_to_uint8_OFFSET


	def find_old_label_ID(self, masked_bi, masked_ID):
		old_label = 0
		old_ID = -self.int8_to_uint8_OFFSET

		unique, counts = np.unique(masked_bi, return_counts=True)
		BI_cnt = dict(zip(unique, counts))
		sorted_dict = sorted(BI_cnt.items(), key=operator.itemgetter(1))
		if sorted_dict[-1][0] == 0:
			if len(sorted_dict) > 1:
				old_label = sorted_dict[-2][0]
			else:
				old_label = 0

		uniqueID, countsID = np.unique(masked_ID, return_counts=True)
		ID_cnt = dict(zip(uniqueID, countsID))
		sorted_dict_ID = sorted(ID_cnt.items(), key=operator.itemgetter(1))
		if sorted_dict_ID[-1][0] == -self.int8_to_uint8_OFFSET:
			if len(sorted_dict_ID) > 1:
				old_ID = sorted_dict_ID[-2][0]
			else:
				old_ID = -self.int8_to_uint8_OFFSET		

		return (old_label, old_ID)


	# def find_old_label(self, rect):
	# 	final_BI_rect = self.final_BI[rect[0] : rect[1], rect[2] : rect[3]]
	# 	unique, counts = np.unique(final_BI_rect, return_counts=True)
	# 	BI_cnt = dict(zip(unique, counts))
	# 	## sort by value; ascending order
	# 	sorted_dict = sorted(BI_cnt.items(), key=operator.itemgetter(1))

	# 	if sorted_dict[-1][0] == 0:
	# 		if len(sorted_dict) > 1:
	# 			return sorted_dict[-2][0]
	# 		else:
	# 			return 0


	def _ID2label_dict_add(self, ID, label):
		"""
		add pair ID - label
		"""
		print "@@@@@@@@@@@ [_ID2label_dict_add]>> ID: {}; label: {}".format(ID, label)
		print "1 self.line_ID2label: ", self.line_ID2label
		unique, counts = np.unique(self.final_ID, return_counts=True)
		unique_label, counts_label = np.unique(self.final_BI, return_counts=True)
		final_ID_cnt = dict(zip(unique, counts))
		final_BI_cnt = dict(zip(unique_label, counts_label))
		print "final_ID_cnt>>>>>>>: ", final_ID_cnt
		print "final_BI_cnt>>>>>>>: ", final_BI_cnt

		if ID not in final_ID_cnt.keys():
			print ">>> try to add pair, but ID: {} not in final_ID".format(ID)
			return

		if label not in final_BI_cnt.keys():
			print ">>> try to add pair, but label: {} not in final_BI".format(label)
			return

		if ID not in self.line_ID2label.keys():
			if ID not in final_ID_cnt.keys():
				print ">>> try to add pair, but ID: {} not in final_ID".format(ID)
			else:
				self.line_ID2label[ID] = [label]
		elif label not in self.line_ID2label[ID]:
			if label not in final_BI_cnt.keys():
				print ">>> try to add pair, but label: {} not in final_BI".format(label)
			else:
				self.line_ID2label[ID].append(label)

		print "2 >>>>>>>>>>>>>>>>>>>: ", self.line_ID2label


	def _ID2label_dict_del(self, ID, label):
		"""
		del pair ID - label
		"""
		print "@@@@@@@@@@@ [_ID2label_dict_del]>> ID: {}; label: {}".format(ID, label)
		print "1 self.line_ID2label: ", self.line_ID2label
		unique, counts = np.unique(self.final_ID, return_counts=True)
		unique_label, counts_label = np.unique(self.final_BI, return_counts=True)
		final_ID_cnt = dict(zip(unique, counts))
		final_BI_cnt = dict(zip(unique_label, counts_label))
		print "final_ID_cnt>>>>>>>: ", final_ID_cnt
		print "final_BI_cnt>>>>>>>: ", final_BI_cnt

		if ID in self.line_ID2label.keys() and len(self.line_ID2label[ID]) > 1:
			self.line_ID2label[ID].remove(label)
			return

		if ID in self.line_ID2label.keys() and len(self.line_ID2label[ID]) == 1:
			if ID in final_ID_cnt.keys():
				print ">>> try to del pair, but ID: {} still in final_ID".format(ID)
			else:
				del self.line_ID2label[ID]

		print "2 >>>>>>>>>>>>>>>>>>>: ", self.line_ID2label


	# def _update_ID2label(self, ID, label):
	# 	"""
	# 	update self.line_ID2label according to self.final_ID
	# 	assuming we allow one ID has more than one paired labels
	# 	Case 1. append new label for ID
	# 	Case 2. del ID-label pair
	# 	"""
	# 	print "@@@@@@@@@@@ [update_ID2label]>> ID: {}; label: {}".format(ID, label)
	# 	print "1 self.line_ID2label: ", self.line_ID2label
	# 	unique, counts = np.unique(self.final_ID, return_counts=True)
	# 	unique_label, counts_label = np.unique(self.final_BI, return_counts=True)
	# 	final_ID_cnt = dict(zip(unique, counts))
	# 	final_BI_cnt = dict(zip(unique_label, counts_label))
	# 	print "final_ID_cnt>>>>>>>: ", final_ID_cnt
	# 	print "final_BI_cnt>>>>>>>: ", final_BI_cnt

	# 	## 1. ID in final_ID, append ID-label
	# 	if (ID in final_ID_cnt.keys()) and (final_ID_cnt[ID] != 0) :
	# 		if ID not in self.line_ID2label.keys():
	# 			self.line_ID2label[ID] = [label]
	# 		else:
	# 			old_label_list = self.line_ID2label[ID]
	# 			if label not in old_label_list:
	# 				self.line_ID2label[ID].append(label)
	# 	## 2. ID not in final_ID, del ID-label pair
	# 	else:
	# 		if ID in self.line_ID2label.keys():
	# 			del self.line_ID2label[ID]

	# 	## 1. label in final_BI, append ID-label
	# 	if (label in final_BI_cnt.keys()) and (final_BI_cnt[label] != 0) :
	# 		if (ID in self.line_ID2label.keys()) and (label not in self.line_ID2label[ID]):
	# 			self.line_ID2label[ID].append(label)
	# 	## 2. label not in final_BI, del ID-label pair
	# 	else:
	# 		if ID in self.line_ID2label.keys():
	# 			del self.line_ID2label[ID]

	# 	# if ID not in final_ID_cnt.keys() or final_ID_cnt[ID] == 0:
	# 	# 	if ID in self.line_ID2label.keys():
	# 	# 		# print ">>>>>>>> del ID: ", ID
	# 	# 		del self.line_ID2label[ID]
	# 	# elif ID not in self.line_ID2label.keys():
	# 	# 	self.line_ID2label[ID] = label

	# 	print "2>>>>>>>>>>>>>>>>>>>: ", self.line_ID2label



	def show_as_biImg(self, bi):
		tmp = np.zeros(bi.shape, np.uint8)
		tmp[ bi != 0 ] = 255

		biImg = Image.fromarray(tmp)
		biImg.show()

	def update_polygon_value(self):
		"""
				1) draw polygon according to seg_arr and label (color)
				2）reset seg_arr, collect_poly_points
		"""
		edit_method = self.parent().edit_method
		line2_mode = self.parent().line2_mode if edit_method == "line2" else None

		if line2_mode == "polygonMinus":
			label = 0
		elif line2_mode == "polygonPlus":
			label = self.cur_line_label

		if self.Zoomed == True:
			for i, point in enumerate(self.zoomed_collect_points):
				if i == 0 and len(self.collect_poly_points) > 0:
					print "skip 1st point in zoomed_collect_points"
					continue
				else:
					pointS = self._point_large2small([point])
					# print "pointS = ", pointS				
					self.collect_poly_points.extend(pointS)

		collect_points = [tuple(p) for p in self.collect_poly_points]
		# bi_arr = self.lineSeg_BI_dict[self.cur_line_type2][1] if self.cur_line_type2 in self.lineSeg_BI_dict.keys() else np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
		if self.final_BI is None:
			self.final_BI = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
		bi_arr = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
		bi_arr[self.final_BI == self.cur_line_label] = self.cur_line_label
		# bi_arr[bi_arr != 0] = 255
		# Image.fromarray(bi_arr).show()

		## boundingRect of polygon (in original_sized img)
		rx, ry, rw, rh = cv2.boundingRect(np.array(collect_points))
		rect = [ry, ry + rh, rx, rx + rw]

		### Finding old label and old ID ###
		polygonMask = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
		polygonMask_img = Image.fromarray(polygonMask, 'L')
		mask_draw = ImageDraw.Draw(polygonMask_img)
		mask_draw.polygon(collect_points, fill=255, outline=255)
		polygonMask = np.array(polygonMask_img)

		masked_bi = bi_arr.copy()
		masked_bi[polygonMask == 0] = 0

		masked_ID = self.final_ID.copy()
		masked_ID[polygonMask == 0] = -self.int8_to_uint8_OFFSET
		old_label, old_ID = self.find_old_label_ID(masked_bi, masked_ID)
		print ">>>> old_label, old_ID: {}".format(old_label, old_ID)
		###

		if line2_mode == "polygonOtsu":
			# img within rect area
			# img_rect = self.ref_pic.img_arr[ry : (ry + rh), rx : (rx + rw), :]
			## thresholding on original_sized_array
			img_arr = self.ori_img.copy()
			img_rect = img_arr[ry : (ry + rh), rx : (rx + rw), :]
			img_rect_gray = cv2.cvtColor(img_rect, cv2.COLOR_BGR2GRAY)

			# 2. get polygon ROI 
			roi_mask = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8) 
			mask_img = Image.fromarray(roi_mask, 'L')
			draw = ImageDraw.Draw(mask_img)
			draw.polygon(collect_points, fill=255, outline=255)
			roi_mask = np.array(mask_img)
			# crop to rect area
			roi_mask_cropped = roi_mask[ry : (ry + rh), rx : (rx + rw)]

			# 3. get roi row array
			roi_array = img_rect_gray[roi_mask_cropped != 0]
			blur = cv2.GaussianBlur(roi_array, (7, 7), 0)
			ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			# print "th.shape: {}".format(th.shape)
			# print "roi_mask_cropped.shape: {}".format(roi_mask_cropped.shape)
			# print "img_rect_gray.shape: {}".format(img_rect_gray.shape)
			# print "roi_array.shape: {}".format(roi_array.shape)
			# print "(roi_mask_cropped != 0).shape: {}".format((roi_mask_cropped != 0).shape)
			roi_mask_cropped[roi_mask_cropped != 0] = th[:, 0]
			roi_mask_cropped[roi_mask_cropped != 0] = self.cur_line_label
			# Image.fromarray(roi_mask_cropped).show()
			rect_seg = roi_mask_cropped
			newID = self.cur_line_ID
			newLabel = self.cur_line_label


		elif line2_mode == "polygonPlus" or line2_mode == "polygonMinus":
			# img_BI = Image.fromarray(bi_arr, 'L')
			# draw = ImageDraw.Draw(img_BI)
			# collect_points = [tuple(p) for p in self.collect_poly_points]
			# draw.polygon(collect_points, fill=label, outline=label)
			# bi_arr = np.array(img_BI)

			## update lineSeg_BI
			if line2_mode == "polygonPlus":
				tmp_arr = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
			elif line2_mode == "polygonMinus":
				tmp_arr = bi_arr
			tmp_img = Image.fromarray(tmp_arr, 'L')
			draw = ImageDraw.Draw(tmp_img)
			# collect_points = [tuple(p) for p in self.collect_poly_points]
			draw.polygon(collect_points, fill=label, outline=label)
			tmp_arr = np.array(tmp_img)
			rect_seg = tmp_arr[ry : (ry + rh), rx : (rx + rw)]
			newID = self.cur_line_ID
			newLabel = self.cur_line_label

		elif line2_mode == "select":
			typeList = [(t + " | " + self.line_types[t]['chinese']) for t in self.line_types.keys() if "label" in self.line_types[t].keys()]
			curID = old_ID
			# print "line_ID2label[curID][0]>>>>>>>>>>> {}".format(self.line_ID2label[curID][0])
			# print "typeList: {}".format(typeList)
			curType = typeList[self.line_ID2label[curID][0]-1]
			select_type, select_ID, ok = modifyTypeID_Dialog.getTypeAndID(typeList, curType, curID)
			select_type = str(select_type)
			idx = select_type.index(" | ")
			select_type = select_type[:idx]
			select_ID = int(select_ID)
			print "Selected lineType: {}; Selected ID: {}".format(select_type, select_ID)
			if ok == False:
				self.collect_poly_points = []
				return True

			select_label = self.line_types[select_type]["label"]
			
			tmp_arr = bi_arr
			rect_seg = tmp_arr[ry : (ry + rh), rx : (rx + rw)]
			rect_seg[rect_seg != 0] = 255
			Image.fromarray(rect_seg).show()
			newID = select_ID
			newLabel = select_label


		if self.Zoomed == True:
			self.zoomed_collect_points = []

		# Image.fromarray(rect_seg).show()
		# print ">>>>>>>>>>>> rect: ", rect
		# print ">>>>>>>>>>>> rect_seg.shape: ", rect_seg.shape

		# ### TEST rect isCorrect
		# img_arr = self.ori_img.copy()
		# img_crop = img_arr[rect[0] : rect[1], rect[2] : rect[3]]
		# Image.fromarray(img_crop).show()
		
		self.update_lineSeg_BI(rect_seg, rect, old_label, newLabel, old_ID, newID)
		self.collect_poly_points = []


	def check_ID2label_llegal(self):
		for ID in self.line_ID2label.keys():
			if len(self.line_ID2label[ID]) > 1:
				log_str = "[ID2label Error] line_ID2label: {}\n".format(self.line_ID2label)
				self.parent().add_logging(log_str)
				return (ID, False)

		return (0, True)



	def set_ref_img(self, img_arr):
		# seg_arr (r * c) 2D size, init with 255
		# seg_arr[r, c] -- label of pixel on (r, c)
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
			self.seg_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)
			self.seg_arr[:] = 255


			self.final_BI = np.zeros(img_arr.shape[:2], dtype=np.uint8)
			self.final_ID = np.zeros(img_arr.shape[:2], dtype=np.int8)
			self.final_ID[:] = -128
			self.line_ID2label = {}
			self.selected_lineSeg_indices = []
			self.selected_lineSeg_rect = []    # rect: [min_r, max_r, min_c, max_c]

			self.parent().sp.setValue(self.cur_line_ID)

			self.lineSegDist_dict = {}


			# init line_dict
			self.line_set = set()

			# self.final_BI = np.zeros(self.ori_img.shape[:2], np.uint8)

			######
			# init all variables used in "cursor-snap2mid"
			self.mousePaint_layer = self.ref_pic.img_arr
			self.w = self.ref_pic.img_arr.shape[1]
			self.h = self.ref_pic.img_arr.shape[0]
			self.MannualMode = False
			self.h0 = self.h/8
			self.w0 = self.w/8
			# for Linear Shrink
			self.hDelta = 8
			self.wDelta = 14
			# # for k^2 shrink
			# self.hRate = 0.85
			# self.wRate = 0.9
			# H, W for detecting rect of points where mouse cursor currently points
			self.rectH = self.h0
			self.rectW = self.w0
			self.edge_x = []	# left and right point's x on this row's edge [smaller, lager]

			## Linear Shrink
			self.hLookup2 = [self.h-1]
			interval = self.h0
			while (interval > 8 and self.hLookup2[-1] > 0):
				self.hLookup2.append(self.hLookup2[-1]-interval)
				interval -= self.hDelta
			######

			self.update_disp()
			return True
		return False


	# def setCanvas(self, qImg):
	# 	"""
	# 	called when process_anno after set_ref_pic
	# 	"""
	# 	self.canvas.setPixmap(qImg)


	def do_segmentation(self):
		# Apply segmentation and update the display
		"""
		1) Do pre-segmentation (2 methods available)
		2) seg_index -- init with pre-segmentation result
		3) seg_disp -- each pixel is colored according to its class label
		4.1) img_arr -- on ori_img, color each pixel according to seg_disp
		4.2) img_arr -- mark boundaries of segmentations according to seg_index
		"""

		ori_img = self.ref_pic.img_arr
		sp = self.seg_params

		if self.seg_method == 'slic':
			n_segments, compactness, sigma = np.int(
				sp[0]), np.int(sp[1]), sp[2]
			self.seg_index = slic(
				ori_img,
				n_segments=n_segments,
				compactness=compactness,
				sigma=sigma)

		elif self.seg_method == 'felzenszwalb':
			scale, min_size, sigma = np.int(sp[0]), np.int(sp[1]), sp[2]
			self.seg_index = felzenszwalb(
				ori_img, scale=scale, min_size=min_size, sigma=sigma)

		r, c = self.seg_arr.shape
		# color_map -- one row; color_map[2] -- color of class label_2
		# seg_disp -- 3D (r*c*3); seg_disp[r, c] -- color of pixel (r, c) on
		# img according to its label
		self.seg_disp = self.color_map[self.seg_arr.ravel()].reshape((r, c, 3))
		# img_arr -- color the ori_img with the result of pixelwise labeling
		self.img_arr = np.array(
			ori_img * (1 - self.alpha) + self.alpha * self.seg_disp, dtype=np.uint8)
		self.img_arr = np.array(
			mark_boundaries(
				self.img_arr,
				self.seg_index) * 255,
			dtype=np.uint8)
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

		# 2) load _LaneLineBI.png, _LaneLineID.png and json
		lineBI_path = seg_path[:-8] + "_LaneLineBI.png"
		lineID_path = seg_path[:-8] + "_LaneLineID.png"
		if os.path.exists(lineBI_path):
			lineBI_arr = np.array(Image.open(lineBI_path))
		else:
			print "No laneLineBI.png exisits! Creating empty numpy ndarray..."
			lineBI_arr = np.zeros(ori_img.shape[:2], np.uint8)
			# Image.fromarray(lineBI_arr, 'L').save(lineBI_path)

		if os.path.exists(lineID_path):
			lineID_arr = np.array(Image.open(lineID_path))
		else:
			print "No laneLineID.png exisits! Creating empty numpy ndarray..."
			lineID_arr = np.zeros(ori_img.shape[:2], np.uint8)
			# Image.fromarray(lineBI_arr, 'L').save(lineBI_path)

		assert(len(lineBI_arr.shape) == 2)
		self.final_BI = lineBI_arr
		self.final_ID = uint8_to_int8(lineID_arr, self.int8_to_uint8_OFFSET)
		# tmp = np.zeros(self.final_ID.shape, np.uint8)
		# tmp[self.final_ID == 0] = 255
		# Image.fromarray(tmp).show()

		# print "@@@@@@@@@@@################### ", np.count_nonzero(self.final_BI)
		# final_BI = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
		# final_BI[self.final_BI != 0] = 255
		# Image.fromarray(final_BI).show()	
		

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


	# def load_segimage(self, seg_path):
	# 	# load and display a segmented img
	# 	# 1) seg_path -- e.g. 00001_seg.png  --loads to--> self.seg_arr
	# 	# 2） load 0001_seg.json 'lines' --> self.line_dict

	# 	# 1) load Segmentation
	# 	seg = np.array(Image.open(seg_path))
	# 	# seg_arr is 2D r*c
	# 	assert(len(seg.shape) == 2)
	# 	self.seg_arr = seg

	# 	# print set(self.seg_arr.ravel()), 'seg arr sum'
	# 	assert(not (self.ref_pic is None))
	# 	ori_img = self.ref_pic.img_arr
	# 	# if ori_img.shape[:2] != self.seg_arr.shape[:2]:
	# 	assert(ori_img.shape[:2] == self.seg_arr.shape[:2])

	# 	# 2) load _LaneLineSeg.png and json
	# 	lineSeg_path = seg_path[:-8] + "_LaneLineSeg.png"
	# 	if os.path.exists(lineSeg_path):
	# 		lineSeg_arr = np.array(Image.open(lineSeg_path), 'L')
	# 	else:
	# 		print "No SegBI {} exisits! Creating empty png..."
	# 		lineSeg_arr = np.zeros(ori_img.shape[:2], np.uint8)
	# 		Image.fromarray(lineSeg_arr, 'L').save(lineSeg_path)

	# 	assert(len(lineSeg_arr.shape) == 2)
		

	# 	_, ext = os.path.splitext(seg_path)
	# 	seg_meta_path = seg_path.replace(ext, '.json')

	# 	# print "seg_meta_path: ", seg_meta_path

	# 	if os.path.exists(seg_meta_path):
	# 		print "Loading seg.png and .json file from {}...".format(seg_meta_path)
	# 		d = read_json(seg_meta_path)
	# 		print "read_json: ", seg_meta_path
	# 		if ('lines' in d.keys()):
	# 			tmp_list = d['lines']
	# 			for tmp in tmp_list:
	# 				loadline = self.LaneLine()
	# 				loadline.line_type = tmp['line_type']
	# 				# loadline.points[:] = tmp['points']
	# 				points = tmp['points']
	# 				for p in points:
	# 					loadline.points.append([int(p[0]), int(p[1])])
	# 				# loadline.break_idx[:] = tmp['break_idx']
	# 				break_i = tmp['break_idx']
	# 				for i in break_i: 
	# 					loadline.break_idx.append(int(i))
	# 				self.line_set.add(loadline)
	# 			print self.line_set

	# 		if ('LaneLineSeg_info' in d.keys()):
	# 			for label in d["LaneLineSeg_info"].keys():
	# 				t = d["LaneLineSeg_info"][label]
	# 				print "label: {}; t: {}".format(label, t)
	# 				self.lineSeg_BI_dict[t] = []
	# 				BiBlock_stk = []
	# 				final_BI = np.zeros(lineSeg_arr.shape, np.uint8)
	# 				# print "lineSeg_arr shape: {}".format(lineSeg_arr.shape)
	# 				# print "lineSeg_arr == label: {}".format(lineSeg_arr == int(label))
	# 				final_BI[lineSeg_arr == int(label)] = label

	# 				self.lineSeg_BI_dict[t].append(BiBlock_stk)
	# 				self.lineSeg_BI_dict[t].append(final_BI)

	# 				## load lineSegDisp_dict (must after self.lineSeg_BI_dict[t] updated)
	# 				self.lineSegDisp_dict[t] = self._get_lineSeg_Disp(t)


	# 	self.update_disp()
	# 	self.update()

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
		## Check if illegal
		ID, legal = self.check_ID2label_llegal()
		if legal == False:
			self.show_conflict_msgBox(ID)
			return False
		# 1) save seg_arr into .png file
		Image.fromarray(np.array(self.seg_arr, dtype=np.uint8)).save(save_path)

		# 2) save final_BI into xxxx_LaneLineBI.png, final_ID into xxxx_LaneLineID.png file
		lineBI_save_path = save_path[:-8] + "_LaneLineBI.png"
		lineID_save_path = save_path[:-8] + "_LaneLineID.png"
		# lineBI_arr = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
		# for t in sorted(self.lineSeg_BI_dict.iterkeys()):
		# final_BI = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
		# final_BI[self.final_BI != 0] = 255
		# Image.fromarray(final_BI).show()
		Image.fromarray(self.final_BI).save(lineBI_save_path)
		lineID_arr = int8_to_uint8(self.final_ID, self.int8_to_uint8_OFFSET)
		Image.fromarray(lineID_arr).save(lineID_save_path)

		# 3) save self.label_type and self.label_names info into json
		_, ext = os.path.splitext(save_path)
		save_meta_path = save_path.replace(ext, '.json')
		self.save_meta(save_meta_path)

		return True


	# def save(self, save_path):
	# 	# print "*@@@@@@@@@@@@@saving@@@@@@@@@@@@@"
	# 	# save_path: "{}/{}_seg.png".format(self.anno_dir, base_name)
	# 	# 1) save seg_arr into .png file
	# 	Image.fromarray(np.array(self.seg_arr, dtype=np.uint8)).save(save_path)

	# 	# 2) save line_segBI into xxxx_LaneLineSeg.png file
	# 	lineSeg_save_path = save_path[:-8] + "_LaneLineSeg.png"
	# 	lineSeg_arr = np.zeros(self.ref_pic.img_arr.shape[:2], np.uint8)
	# 	# for t in sorted(self.lineSeg_BI_dict.iterkeys()):
	# 	for t in self.line_types.keys():
	# 		if t not in self.lineSeg_BI_dict.keys():
	# 			continue
	# 		print ">>>>>>>>>>>>save: {}".format(t)
	# 		lineSeg_label = self.line_types[t]["label"]
	# 		# print ">>>>>>>>> len(dict): {}".format(len(self.lineSeg_BI_dict[t]))
	# 		final_BI = self.lineSeg_BI_dict[t][1]
	# 		lineSeg_arr[final_BI != 0] = lineSeg_label

	# 	Image.fromarray(lineSeg_arr, 'L').save(lineSeg_save_path)

	# 	# 3) save self.label_type and self.label_names info into json
	# 	_, ext = os.path.splitext(save_path)
	# 	save_meta_path = save_path.replace(ext, '.json')
	# 	self.save_meta(save_meta_path)


	def save_meta(self, save_path):
		d = OrderedDict()
		if self.parent().edit_method == "segmentation" or self.parent().edit_method == "polygon":
			d['label_type'] = self.label_type
			d['label_names'] = self.label_names
			# print "want to save: ", [line.__dict__ for line in self.line_set]

		## save polyline data
		print ">>>>>>>>>>line_set --> {}".format(self.line_set)
		d['polylines'] = [line.__dict__ for line in self.line_set]

		## save binarized_line data
		# d = dict()
		biLine_dict = dict()
		d["binarized_line"] = biLine_dict

		## 1) save label-type key-value paires in field "label_type"
		labelType_dict = dict()
		biLine_dict["label_type"] = labelType_dict

		for t in self.line_types.keys():
			if "label" not in self.line_types[t].keys():
				continue
			label = self.line_types[t]["label"]
			if label in self.final_BI:
				labelType_dict[label] = t
		# print ">>>>>>>>>>>>>>> ", biLine_dict

		biLine_dict["label_type"] = labelType_dict

		## 2) save ID-label key-value paires in field "ID_label"
		biLine_dict["ID_label"] = self.line_ID2label
		biLine_dict["LaneLineID_png_offset"] = self.int8_to_uint8_OFFSET

		d["binarized_line"] = biLine_dict

		if len(self.line_set) != 0 or len(labelType_dict) != 0:
			self.skip = False
		d["skip"] = 1 if self.skip == True else 0

		write_json(save_path, d)		


	# def save_meta(self, save_path):
	# 	d = OrderedDict()
	# 	d['label_type'] = self.label_type
	# 	d['label_names'] = self.label_names
	# 	# print "want to save: ", [line.__dict__ for line in self.line_set]

	# 	# print ">>>>>>>>>>line_set2 --> {}".format(self.line_set)
	# 	if self.parent().edit_method == "line":
	# 		d['lines'] = [line.__dict__ for line in self.line_set]

	# 	if self.parent().edit_method == "line2":
	# 		lineSeg_info = dict()
	# 		# for t in sorted(self.lineSeg_BI_dict.iterkeys()):
	# 		for t in self.line_types.keys():
	# 			if t not in self.lineSeg_BI_dict.keys():
	# 				continue
	# 			final_BI = self.lineSeg_BI_dict[t][1]
	# 			if np.count_nonzero(final_BI) != 0:
	# 				lineSeg_info[self.line_types[t]["label"]] = t

	# 		d['LaneLineSeg_info'] = lineSeg_info

	# 	# print "ready to save d: ", d
	# 	## In 'line2' mode, save all line segamentation binary image with lineType

	# 	write_json(save_path, d)

	def checkpoint(self):
		"""
		save current segmentation state
		seg method:
		seg parameter:
		index_map
		"""
		pass