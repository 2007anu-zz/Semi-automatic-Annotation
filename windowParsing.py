#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Annotating Window for Parsing
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
from PIL import ImageFont
import cv2

from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries

import scipy.io as sio

import functools

from segimg_parsing import SegPic
from my_pic import MyPic
# from infoLabel import InfoLabel
# imgproc part


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
##


import logging


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


class Worker(QtCore.QThread):
	def __init__(self, parent, work_name, signal_name):
		QtCore.QThread.__init__(self, parent=parent)
		self.signal = QtCore.SIGNAL(signal_name)

	def run(self):
		while(True):
			time.sleep(0.1)
			self.emit(self.signal, "hi from thread")


class ParsingWindow(QtGui.QWidget):
	__version = '0.18'
	__release_date = '2017-07-14'

	def appinit(self):
		thread = Worker(self, 'play_worker', 'play')
		self.connect(thread, thread.signal, self.play)
		thread.start()

	def set_default_config(self):
		self.config = OrderedDict([('car',
									OrderedDict(['label_type',
												 'cityscape_exp_1'])),
								   ('drone',
									OrderedDict(['label_type',
												 '18_category_exp_0']),
									('seg_method',
									 'contour')
									)
								   ])

	def load_config(self):
		self.mydir = os.path.dirname(os.path.realpath(__file__))
		fp = self.mydir + '/' + '.anno_init.json'
		if os.path.exists(fp):
			self.config = read_json(fp)
			print 'Load config succussfully'
		else:
			self.set_default_config()

	def __init__(self, *args):
		"""

		tab_widget --- self.tab_drone
										 ---drone_main_vertical
						   --- self.tab_car
										 ---car_main_vertical
		drone_main_vertical
						   --- Hlayout_radio_drone

		car_main_vertical
						   --- Hlayout_ratio_car

		Hlayout

		tab_widget_method
						   ---  tab_method_slider
						   ---  tab_method_polygon
						   ---  tab_method_line

		"""
		self.TASK = 'Parsing'

		self.edit_method = 'segmentation'
		self.select_line = False  # Line selecting state

		# [Key Mode]
		self.exit_select_line_mode = False
		self.cutline_mode = False
		self.bbox_mode = False
		self.posi_point_mode = True

		QtGui.QWidget.__init__(self, *args)
		self.load_config()

		self.seg_method = self.config['seg_method']

		self.setWindowTitle('Annotation - Parsing')
		self.tab_widget = QtGui.QTabWidget()
		self.tab_drone = QtGui.QWidget()
		self.tab_car = QtGui.QWidget()
		drone_main_vertical = QtGui.QVBoxLayout(self.tab_drone)
		car_main_vertical = QtGui.QVBoxLayout(self.tab_car)

		self.tab_widget.addTab(self.tab_drone, 'Drone')
		self.tab_widget.addTab(self.tab_car, 'Car')
		print 'Current index {}'.format(self.tab_widget.currentIndex())
		self.tab_widget.currentChanged.connect(self.tab_onchange)
		self.tab_widget.setCurrentIndex(1)

		self.mode = 'edit'
		self.ShowSkipImg = False   # show skip init to False

		self.mode_checkbox = QtGui.QCheckBox(u'编辑模式 (Edit)', self)
		self.mode_checkbox.setChecked(True)
		self.mode_checkbox.stateChanged.connect(self.mode_change)

		self.showSkip_checkbox = QtGui.QCheckBox(u'显示跳过文件 (Show Skipped)', self)
		self.showSkip_checkbox.setChecked(False)
		self.showSkip_checkbox.stateChanged.connect(self.showSkip_change)

		img_disp_W = 640
		img_disp_H = 480
		W_free = 300
		H_free = 300
		WinWidth = img_disp_W * 2 + W_free
		WinHeight = img_disp_H + H_free
		self.WinWidth = WinWidth
		self.WinHeight = WinHeight

		self.setGeometry(200, 200, WinWidth, WinHeight)
		# create objects
		# original img -- on the LEFt
		self.label_img = MyPic(self)
		self.label_img.set_main_window(self)
		self.label_img.setScaledContents(False)
		self.label_img.setAlignment(QtCore.Qt.AlignLeft)

		self.label_img.setGeometry(10, 10, img_disp_W, img_disp_H)
		self.label_img.read_image(None)
		self.label_img.setPixmap(
			QtGui.QPixmap(
				cvtrgb2qtimage(
					self.label_img.img_arr)))

		# segmented img -- on the RIGHT
		self.label_seg = SegPic(self)
		self.label_seg.set_seg_config(self.label_img,
									  self.config['drone']['label_type'],
									  self.seg_method)
		self.label_seg.setScaledContents(False)
		self.label_seg.setAlignment(QtCore.Qt.AlignLeft)

		self.label_seg.setGeometry(50 + img_disp_W, 10, img_disp_W, img_disp_H)
		self.label_seg.read_image(None)
		self.label_seg.setPixmap(
			QtGui.QPixmap(
				cvtrgb2qtimage(
					self.label_seg.img_arr)))

		## Set Cursor to [Cross Hand]
		# self.label_seg.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
		
		## set defaul tab to -- Car
		self.tab_onchange(1)

		# self.label_seg.setCanvas(
		# 	QtGui.QPixmap(
		# 		cvtrgb2qtimage(
		# 			self.label_seg.img_arr)))

		# self.line_type_list = [t for t in self.label_seg.line_types.keys() if "label" in self.label_seg.line_types[t].keys()]
		# self.line_type_list.remove("polygonPlus")
		# self.line_type_list.remove("polygonMinus")
		# self.line_type_list.remove("polygonOtsu")

		#self.radiobtn_dict = {}     # storing QtRatioBtn for 'line' mode
		#self.radiobtn_dict2 = {}    # storing QtRatioBtn for 'line2' mode

		#### RightClick ContextMenu for <line2> edit_method ####
		# set SegPic context menu policy
		self.label_seg.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
		self.label_seg.customContextMenuRequested.connect(self.on_context_menu)

		"""
		# Create contex Menu
		self.r_menu = QtGui.QMenu(self)  # right click to select an edit mode
		act_group = QtGui.QActionGroup(self, exclusive=True)

		self.polygonOtsuAction = QtGui.QAction(u'自动多边形分割 (Polygon Thresholding)', self, checkable=True)
		self.polygonOtsuAction.triggered.connect(lambda: self.polygonOtsuAction_slot())
		otsu = act_group.addAction(self.polygonOtsuAction)
		self.r_menu.addAction(otsu)

		# self.smartGCutAction = QtGui.QAction('Smart GrabCut', self, checkable=True)
		# self.smartGCutAction.triggered.connect(lambda: self.smartGCutAction_slot())
		# sgc = act_group.addAction(self.smartGCutAction)
		# self.r_menu.addAction(sgc)
	    
		
		self.polygonMinusAction = QtGui.QAction(u'多边形- (Polygon-)', self, checkable=True)
		self.polygonMinusAction.triggered.connect(lambda: self.polygonMinusAction_slot())
		pm = act_group.addAction(self.polygonMinusAction)
		self.r_menu.addAction(pm)

		self.polygonPlusAction = QtGui.QAction(u'多边形+ (Polygon+)', self, checkable=True)
		self.polygonPlusAction.triggered.connect(lambda: self.polygonPlusAction_slot())
		pa = act_group.addAction(self.polygonPlusAction)
		self.r_menu.addAction(pa)
		
		
		self.r_menu.addSeparator()
		
		self.undoSegAction = QtGui.QAction(u'撤销分割 (Undo Seg)', self)
		self.undoSegAction.triggered.connect(lambda: self.undoSegAction_slot())
		self.r_menu.addAction(self.undoSegAction)

		self.undoPointAction = QtGui.QAction(u'撤销点 (Undo Point)', self)
		self.undoPointAction.triggered.connect(lambda: self.undoPointAction_slot())
		self.r_menu.addAction(self.undoPointAction)
		####

		#### <line2> edit_method
		self.line2_mode = True
		"""
		# layout
		layout = QtGui.QHBoxLayout()
		self.Hlayout = layout
		# add label
		self.pic2_Vlayout = QtGui.QVBoxLayout()
		self.pic2_Vlayout.addWidget(self.label_img)

		
		pic2_layout = QtGui.QHBoxLayout()
		pic2_layout.addLayout(self.pic2_Vlayout)
		pic2_layout.addWidget(self.label_seg)
		# pic2_layout.addWidget(self.label_seg.canvas)

		layout.addLayout(pic2_layout)

		# Button zone
		# slider
		# add open directory button
		self.open_button = QtGui.QPushButton(u'打开文件夹 (Open)', self)
		self.open_button.clicked.connect(self.handle_open_button)

		self.open_file_button = QtGui.QPushButton(u'打开文件 (OpenFile)', self)
		self.open_file_button.clicked.connect(self.handle_open_file_button)

		# self.open_button.setGeometry(10,WinHeight -180,100,20)

		# add finish button
		self.save_button = QtGui.QPushButton(u'保存 (save)', self)
		self.save_button.clicked.connect(self.handle_save_button)
		# self.save_button.setGeometry(10,WinHeight-150,100,20)

		self.delete_button = QtGui.QPushButton(u'删除 (delete)', self)
		self.delete_button.clicked.connect(self.handle_delete_button)

		self.skip2Prev_button = QtGui.QPushButton(u'<--跳至前一张 (skip)', self)
		self.skip2Prev_button.clicked.connect(self.handle_skip2Prev_button)

		self.skip2Next_button = QtGui.QPushButton(u'跳至后一张--> (skip)', self)
		self.skip2Next_button.clicked.connect(self.handle_skip2Next_button)

		self.open_dir = None
		self.anno_dir = None
		# Hlayout_radio_drone = QtGui.QHBoxLayout()
		Hlayout_radio_drone = QtGui.QGridLayout()
		Hlayout_radio_car = QtGui.QGridLayout()

		self.tab_widget_method = QtGui.QTabWidget()

		self.tab_method_polygon = QtGui.QWidget()
		self.tab_method_slider = QtGui.QWidget()

		self.tab_widget_method.addTab(self.tab_method_slider, u'Segmentation')
		self.tab_widget_method.addTab(self.tab_method_polygon, u'Polygon')
		self.tab_widget_method.currentChanged.connect(self.tab_method_onchange)

		# self.Hlayout.addWidget(self.tab_widget_method)
		layout.addWidget(self.tab_widget_method)

		# deal with slider
		Vlayout_slider = QtGui.QVBoxLayout(self.tab_method_slider)
		# self.Hlayout.addLayout(Vlayout_slider)
		self.add_slider(Vlayout_slider)

		Vlayout = QtGui.QVBoxLayout(self)
		# Vlayout = QtGui.QVBoxLayout(self.tab_drone)
		self.Vlayout = Vlayout

		# Vlayout.addWidget(self.tab_drone)
		# Vlayout.addWidget(self.tab_car)
		Vlayout.addWidget(self.tab_widget)
		Vlayout.addLayout(self.Hlayout)
		# Vlayout.addWidget(self.tab_widget_method)

		# Vlayout.addLayout(Hlayout_radio)
		drone_main_vertical.addLayout(Hlayout_radio_drone)
		Vlayout.addLayout(drone_main_vertical)

		car_main_vertical.addLayout(Hlayout_radio_car)
		Vlayout.addLayout(car_main_vertical)

		self.add_drone_label_button(
			Hlayout_radio_drone,
			self.config['drone']['label_type'])
		self.add_car_label_button(
			Hlayout_radio_car,
			self.config['car']['label_type'])

		subHlayout = QtGui.QHBoxLayout(self)
		subHlayout.addWidget(self.open_button)
		subHlayout.addWidget(self.open_file_button)
		subHlayout.addWidget(self.save_button)
		subHlayout.addWidget(self.delete_button)
		subHlayout.addWidget(self.skip2Prev_button)
		subHlayout.addWidget(self.skip2Next_button)
		subHlayout.addWidget(self.mode_checkbox)
		subHlayout.addWidget(self.showSkip_checkbox)
		Vlayout.addLayout(subHlayout)

		# Disp log
		self.log_handler = QPlainTextEditLogger(self)
		logging.getLogger().addHandler(self.log_handler)
		self.log_handler.widget.setGeometry(
			200, WinHeight - 200, WinWidth / 2, 180)
		Vlayout.addWidget(self.log_handler.widget)

		self.c = 0
		self.play_idx = 0
		self.edit_status = False
		self.process_idx = 0

		self.connect(self.label_seg, QtCore.SIGNAL("click"), self.update)

		self.images_path = None
		self.image_fid_lookup = None

		logging.warn('Version {}\n'.format(self.__version))
		logging.warn('Release date {}\n'.format(self.__release_date))
		self.process_anno()


	def add_logging(self, str):
		logging.warn(str) 

	"""
	def spinBoxValueChange_handel(self):
		self.label_seg.cur_line_ID = self.sp.value()
		logging.warn("Change cur_line_ID to {}\n".format(self.label_seg.cur_line_ID))
		print "cur_line_ID = {}".format(self.label_seg.cur_line_ID)


	def handle_delete_all_LaneLine_Seg_btn(self):
		win_text = "Delete anno. data"
		questiong_msg = "Are you sure to delete all LaneLine Seg annotation on this image?"
		choice = QtGui.QMessageBox.question(self, win_text, questiong_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

		if choice == QtGui.QMessageBox.Yes:
			self.delete_all_LaneLine_Seg()
		else:
			pass


	def delete_all_LaneLine_Seg(self):
		print "Delete all LaneLine_seg on screen!!!"
		self.label_seg.final_BI = np.zeros(self.label_seg.ref_pic.img_arr.shape[:2], np.uint8)
		self.label_seg.final_ID = np.zeros(self.label_seg.ref_pic.img_arr.shape[:2], np.int8)
		self.label_seg.final_ID[:] = -self.label_seg.int8_to_uint8_OFFSET
		self.label_seg.line_ID2label = {}
		self.label_seg.update_disp()
		self.label_seg.update()
	"""

	def tab_method_onchange(self, idx):
		self.tab_widget_method.setCurrentIndex(idx)
		self.label_seg.collect_poly_points = []
		self.label_seg.zoomed_collect_points = []
		if idx == 0:
			self.edit_method = 'segmentation'
			logging.warn('Changed to use <superpixel>\n')
		elif idx == 1:
			self.edit_method = 'polygon'
			logging.warn('Changed to use <Line2>\n')
		self.label_seg.update_disp()
		self.label_seg.update()


	def mode_change(self, state):
		if state == QtCore.Qt.Checked:
			self.mode = 'edit'
			logging.warn(u'进入【编辑】模式 Enter ---[Edit] mode\n')
		else:
			self.mode = 'view'
			logging.warn(u'进入【只读】模式 Enter ---[view] mode\n')

	def showSkip_change(self, state):
		if state == QtCore.Qt.Checked:
			self.ShowSkipImg = True
			logging.warn(u'显示跳过文件 Show Skipped Images\n')
		else:
			self.ShowSkipImg = False
			logging.warn(u'隐藏跳过文件 Hide Skipped Images\n')		

	def tab_onchange(self, idx):
		# set default mode
		if idx == 0:
			label_type = self.config['drone']['label_type']
			self.label_seg.set_seg_config(
				self.label_img, label_type, self.seg_method)
			self.label_img.disp_size = None 
			self.label_img.update()
		else:
			# ### label for displaying info and time
			# self.Info_label = InfoLabel(self)
			# self.pic2_Vlayout.addWidget(self.InfoLabel)
			# ###

			label_type = self.config['car']['label_type']
			# W, H  ??? In car mode, resize to small size ???
			self.label_img.disp_size = (320, 240)
			self.label_seg.set_seg_config(
				self.label_img, label_type, self.seg_method)
			# W, H   ???? duplicate line ????
			
			self.label_img.update()

	def handle_delete_button(self):
		# when pressing delete, reset seg_arr, line_dict; redisplay image;
		win_text = "Delete anno. data"
		questiong_msg = u"Are you sure to delete ALL segmentation annotation on this image?\n确定删除本图片所有标注？"
		choice = QtGui.QMessageBox.question(self, win_text, questiong_msg, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

		if choice == QtGui.QMessageBox.Yes:
			self.delete_seg()
		else:
			pass

	def handle_skip2Prev_button(self):
		## To Prev Image
		## Mark cur Image -- skip == True
		print "<< Mark skip and to Prev img"
		if self.mode == 'edit':
			self.label_seg.skip = True    # skip this image
			successful = self.handle_save_button()
		else:
			logging.warn(u'不能在视图模式中保存标注 (Can not save in view mode).')
		if successful == True:
			self.process_idx = max(0, self.process_idx - 1)
			self.process_anno(False)
		else:
			logging.warn(u'保存冲突 (Cannot save -- conflict exists in image annotation)')		

	def handle_skip2Next_button(self):
		print "Mark Skip and go to Next >>"
		## To Next Image
		## Mark cur Image -- skip == True
		if self.mode == 'edit':
			self.label_seg.skip = True    # skip this image
			successful = self.handle_save_button()
		else:
			logging.warn(u'不能在视图模式中保存标注 (Can not save in view mode).')
		if successful == True:
			self.process_idx = self.process_idx + 1      # jump to next
			if self.process_idx < len(self.images_path):
				self.process_anno(True)
		else:
			logging.warn(u'保存冲突 (Cannot save -- conflict exists in image annotation)')	

	def delete_seg(self):
		self.label_seg.seg_arr[:] = 255
		self.label_seg.line_dict = dict()
		self.label_seg.update_disp()
		self.label_seg.update()


	def add_car_label_button(self, layout, label_type):

		mydir = os.path.dirname(os.path.realpath(__file__))
		fp = '{}/data/{}_labels_Chinese.txt'.format(mydir, label_type)
		label_names = self.label_seg.read_label_name(fp)
		label_index = range(len(label_names))
		# self.label_names.append('ignore')
		label_names.append(u"忽略")
		label_index.append(255)   # ignore_lable -- 255
		logging.warn('load label names :{}'.format(label_names))

		self.label_radiobtn = []
		max_col = 10   # max num of radiobtn per row
		nrow = (len(label_names) - 1) // max_col + 1
		for i, name in enumerate(label_names):
			raw_s = name.strip(' |\n').split(',')[0]
			b = QtGui.QRadioButton(unicode(raw_s))
			c = label_index[i]
			# b.toggled.connect(lambda :self.label_changed(c))
			b.toggled.connect(functools.partial(self.label_changed, c))
			x = i / max_col
			y = i % max_col
			layout.addWidget(b, x, y)
			self.label_radiobtn.append(b)

	def add_drone_label_button(self, layout, label_type):
		mydir = os.path.dirname(os.path.realpath(__file__))
		fp = '{}/data/{}_labels_Chinese.txt'.format(mydir, label_type)
		label_names = self.label_seg.read_label_name(fp)
		label_index = range(len(label_names))
		# self.label_names.append('ignore')
		label_names.append(u"忽略")
		label_index.append(255)
		logging.warn('load label names :{}'.format(label_names))
		self.label_radiobtn = []
		max_col = 10
		nrow = (len(label_names) - 1) // max_col + 1
		for i, name in enumerate(label_names):
			raw_s = name.strip(' |\n').split(',')[0]
			b = QtGui.QRadioButton(unicode(raw_s))
			c = label_index[i]
			# b.toggled.connect(lambda :self.label_changed(c))
			b.toggled.connect(functools.partial(self.label_changed, c))
			x = i / max_col
			y = i % max_col
			layout.addWidget(b, x, y)
			# layout.addStretch(1)
			self.label_radiobtn.append(b)

	def label_changed(self, cur_label):
		print 'change current label to {}'.format(cur_label)
		self.label_seg.current_label = cur_label

	def process_anno(self, toNext=True):
		"""
				Display, load segmentation file
		"""
		if not self.images_path:
			return
		ntot = len(self.images_path)  # total num of image to be annotated
		if self.process_idx >= ntot:
			return

		#### SKIP IMG ####
		## if image marked 'skip' AND self.ShowSkipImg == False, skip this image (don't show it)
		if self.ShowSkipImg == False:
			tryLoad = True
		else:
			tryLoad = False

		while (tryLoad == True):
			img_path = self.images_path[self.process_idx]
			print "Try loading: {}...".format(img_path)
			base_name = os.path.splitext(os.path.basename(img_path))[0]
			json_path = '{}/{}_seg.json'.format(self.anno_dir, base_name)

			if self.process_idx >= len(self.images_path):
				print "### Reach to the end of {}".format(base_name)
				logging.warn(u'这是最后一张！\n')
				return

			if os.path.exists(json_path):
				d = read_json(json_path)
				if 'skip' in d.keys() and int(d['skip']) == 1:
					print "--skip img: {}".format(img_path)
					if toNext == True:
						# go to next
						self.process_idx += 1
					else:
						# go to prev
						self.process_idx -= 1
						if (self.process_idx < 0):
							self.process_idx += len(self.images_path)
						# self.process_idx = max(0, self.process_idx - 1)
				else:
					tryLoad = False
			else:
				tryLoad = False
		#################

		if (self.process_idx == 0):
			logging.warn(u'这是第一张! \n')


		img_path = self.images_path[self.process_idx]
		img = Image.open(img_path)
		# print "img.size = ", img.size

		img_arr = np.array(img)   # img_arr inited to be the original img

		img_name = os.path.splitext(os.path.basename(img_path))

		# initial ref_pic should be set to img_arr
		self.label_seg.set_ref_img(img_arr, img_path)
		# ### TEST
		# Image.fromarray(self.label_img.img_arr).show()
		# ###
		self.label_img.update()
		logging.warn('<<{0}/{1}>>\n'.format(self.process_idx+1, len(self.images_path)))

		base_name = os.path.splitext(os.path.basename(img_path))[0]
		### check if this img is marked 'skip'
		json_path = '{}/{}_seg.json'.format(self.anno_dir, base_name)
		if os.path.exists(json_path):
			d = read_json(json_path)
			if 'skip' in d.keys() and int(d['skip']) == 1:
				logging.warn("This img: {} is marked <SKIP>".format(img_name))		
		###
		seg_path = '{}/{}_seg.png'.format(self.anno_dir, base_name)

		# load _seg file if already segmented.
		if os.path.isfile(seg_path):
			print 'Ready to load {}'.format(seg_path)
			self.label_seg.load_segimage(seg_path)

		print 'process anno call seg'
		if self.mode == 'edit':
			if self.edit_method == 'segmentation':
				# only display the pre-seg img if on edit && segmentation mode
				self.label_seg.do_segmentation()
			else:
				self.label_seg.update_disp()
				self.label_seg.update()
		else:
			self.label_seg.update_disp()
			self.label_seg.update()
			logging.warn('Skip (not saved). Because I am in view mode\n !!!!!')

	def generate_seg_color_map(self, img_arr):
		return np.tile(img_arr[:, :, None], [1, 1, 3])

	@classmethod
	def get_default_seg_params(cls, seg_method):
		if seg_method == 'slic':
			return OrderedDict([('n_segments', (250, 0, 600, 1)),
								('compactness', (10, 0, 200, 10)),
								('sigma', (1, 0, 200, 50.0))
								])
		elif seg_method == 'felzenszwalb':
			return OrderedDict([('scale', (100, 0, 300, 1)),
								('min_size', (50, 0, 1000, 10)),
								('sigma', (0.5, 0, 200, 50.0))
								])
		elif seg_method == 'contour':
			return OrderedDict([
								('Theshold', (1, 0, 20, 20.0))
								])

		else:
			raise Exception('>>>??? {}'.format(seg_method))

	def add_slider(self, layout):
		seg_params = self.get_default_seg_params(self.seg_method)
		self.sliders = []
		self.sliders_scale = []
		self.sliders_curvalue = []
		offset = 0
		vlist = []
		for k, v in seg_params.items():
			t = QtGui.QLabel(k)
			layout.addWidget(t)
			s = QtGui.QSlider(QtCore.Qt.Horizontal)
			s.setGeometry(
				self.WinWidth -
				120,
				self.WinHeight -
				100 -
				offset,
				100,
				100)
			s.setMinimum(v[1])
			s.setMaximum(v[2])
			s.setValue(np.int(v[0] * v[3]))
			s.setTickPosition(QtGui.QSlider.TicksBelow)
			s.setTickInterval(10)
			# s.valueChanged.connect(self.slider_value_changed)
			s.sliderReleased.connect(self.slider_released)
			self.sliders.append(s)
			self.sliders_scale.append(v[3])
			self.sliders_curvalue.append(v[0])
			layout.addWidget(s)
			offset = offset + 10
			vlist.append(v[0])
		self.label_seg.set_seg_params(vlist)

	def slider_released(self):
		has_change = False
		for i, s in enumerate(self.sliders):
			if self.sliders_curvalue[i] != s.value():
				logging.warn(
					'{}: value = {}\n'.format(
						i, s.value() / self.sliders_scale[i]))
				self.sliders_curvalue[i] = s.value()
				has_change = True
		logging.warn('\n')
		if has_change:
			vlist = [
				x / y for x,
				y in zip(
					self.sliders_curvalue,
					self.sliders_scale)]
			self.label_seg.set_seg_params(vlist)
			self.label_seg.do_segmentation()

	def init_images_path(self):
		if self.open_dir:
			logging.warn('Open dir {}\n'.format(self.open_dir))
			tmp = sorted(
				getfilelist(
					self.open_dir,
					'.*\d+\.(png|jpeg|jpg|bmp|JPEG)'))
			##### regex doesn't allow "1234_resize.jpg"
			# fidlist = [np.int(os.path.splitext(x)[0]) for x in tmp]
			# s_index = np.argsort(fidlist)
			# fidlist = [fidlist[k] for k in s_index]
			# tmp = [tmp[k] for k in s_index]
			# self.image_fid_lookup = dict(zip(fidlist, range(len(fidlist))))
			# print tmp, 'debug'
			# print self.open_dir, 'debug2'
			self.images_path = [os.path.join(
				str(self.open_dir), str(x)) for x in tmp]
			logging.warn('Found {} images\n'.format(len(self.images_path)))
		else:
			self.image_path = []
			self.image_fid_lookup = dict()
			logging.warn('Empty directry')

	def handle_save_button(self):
		if not self.images_path:
			return
		ntot = len(self.images_path)
		if self.process_idx >= ntot:
			return
		img_path = self.images_path[self.process_idx]
		base_name = os.path.splitext(os.path.basename(img_path))[0]

		# save label_seg to _seg.png
		successful = self.label_seg.save('{}/{}_seg.png'.format(self.anno_dir, base_name))
		if successful == True:
			logging.warn('save to {} succussfully\n'.format(
				'{}/{}_seg.png, .json'.format(self.anno_dir, base_name)))
			logging.warn(u'保存标注文件 {} 成功！'.format(base_name))

		return successful

	def handle_open_button(self):
		self.open_dir = QtGui.QFileDialog.getExistingDirectory(
			None, 'Select a folder:', '/media/sijin', QtGui.QFileDialog.ShowDirsOnly)
		self.init_images_path()
		if self.open_dir:
			self.anno_dir = self.open_dir + '_' + self.TASK + '_anno'
			ensure_dir(str(self.anno_dir))
		self.process_anno()

	def set_process_idx(self, st):
		try:
			images_fn = [os.path.basename(x) for x in self.images_path]
			self.process_idx = images_fn.index(os.path.basename(st))
		except BaseException:
			print st, images_fn
			pass

	def handle_open_file_button(self):
		file_path = QtGui.QFileDialog.getOpenFileName(self, 'select a file')
		self.open_dir = os.path.dirname(str(file_path))
		self.init_images_path()
		self.set_process_idx(str(file_path))
		if self.open_dir:
			self.anno_dir = self.open_dir + '_' + self.TASK + '_anno'
			ensure_dir(str(self.anno_dir))
		self.process_anno()

	def handle_play_button(self):
		# not in use currently
		self.play_status = not self.play_status
		if self.play_status:
			self.emit(QtCore.SIGNAL("play"))
		print 'Current play status = {}'.format(self.play_status)

	def read_xml(self, xml_path):
		return read_xml(xml_path)

	def add_id(self, anno, XXX=None):
		pass

	def play(self):
		# In edit mode
		mode = 'edit'
		if mode == 'edit':
			pass

	def update(self):
		print 'In main window update'
		pass

	# def wheelEvent(self, event):

	# 	if self.edit_method != 'line' and self.edit_method != 'line2':
	# 		print "[wheelEvent]: Not in line/line2 mode"
	# 		return

	# 	idx_delta = 1
	# 	if self.edit_method == "line":
	# 		type_idx = self.line_type_list.index(self.label_seg.cur_line_type)

	# 	elif self.edit_method == "line2":
	# 		type_idx = self.line_type_list.index(self.label_seg.cur_line_type2)

	# 	if (event.delta() < 0):
	# 		# if self.edit_method == "line" and self.label_seg.cur_line_type == "reversible":
	# 		# 	idx_delta = 4
	# 		# elif self.edit_method == "line2" and self.label_seg.cur_line_type == "reversible":
	# 		# 	idx_delta = 4
	# 		type_idx = (type_idx + idx_delta) % len(self.line_type_list)
	# 		# print "len of line_type_list: {}; type_idx: {};".format(len(self.line_type_list), type_idx)
	# 		self.line_type_changed(self.line_type_list[type_idx])
	# 	else:
	# 		# if self.edit_method == "line" and self.label_seg.cur_line_type == "dash_single_white":
	# 		# 	print "$$$$$$$$$$$$$$$$$$$$"
	# 		# 	idx_delta = 4
	# 		# elif self.edit_method == "line2" and self.label_seg.cur_line_type == "dash_single_white":
	# 		# 	print "$$$$$$$$$$$$$$$$$$$$##########"
	# 		# 	idx_delta = 4
	# 		type_idx -= idx_delta
	# 		if (type_idx < 0):
	# 			type_idx += len(self.line_type_list)
	# 		self.line_type_changed(self.line_type_list[type_idx])

	# 	# print "type_idx: ", type_idx
	# 	# print "line_type: ", self.line_type_list[type_idx] 

	# 	if self.edit_method == "line":
	# 		self.radiobtn_dict[self.line_type_list[type_idx]].setChecked(True)
	# 	if self.edit_method == "line2":
	# 		self.radiobtn_dict2[self.line_type_list[type_idx]].setChecked(True)
	# 	print event.delta()

	def keyReleaseEvent(self, event):
		if event.isAutoRepeat():
			return

		if event.key() == QtCore.Qt.Key_V:
			print "[Mode - Mannual Cursor] OFF"
			self.label_seg.MannualMode = False
		if (event.key() == QtCore.Qt.Key_Z or event.key() == QtCore.Qt.Key_W) and self.edit_method != "segmentation":
			self.label_seg.zoom_out()
			print "[Mode - Zoom OFF] "
		if event.key() == QtCore.Qt.Key_G and (self.edit_method == "line" or self.edit_method == "line2"):
			# self.label_seg.MannualMode = False
			self.line2_mode = "polygonOtsu"
			self.select_line = False
			print "[Mode - select_line]: ", self.select_line
		if event.key() == QtCore.Qt.Key_C and self.edit_method == "line":
			self.cutline_mode = False
			print "[Mode - cutline]: ", self.cutline_mode
		if (event.key() == QtCore.Qt.Key_B or event.key() == QtCore.Qt.Key_N or event.key() == QtCore.Qt.Key_M):
			self.label_seg.disp_segBI_off()

	def keyPressEvent(self, event):
		# print 'I am in keypress event'
		# print event.key()
		k = event.key()
		# print k == QtCore.Qt.Key_F2

		if k == QtCore.Qt.Key_Delete:
			self.handle_delete_button()
			return
		if k < 0 or k > 255:
			return

		if event.key() == QtCore.Qt.Key_V:
			if event.isAutoRepeat():
				return
			self.label_seg.MannualMode = True
			print "[Mode - Mannual Cursor]: ", self.MannualMode

		if (event.key() == QtCore.Qt.Key_Z or event.key() == QtCore.Qt.Key_W) and self.edit_method != "segmentation":
			if event.isAutoRepeat():
				return
			else:
				self.zoomed_BI = None
			self.label_seg.zoom_in()
			print "[Mode - Zoom In] "

		if event.key() == QtCore.Qt.Key_G and (self.edit_method == "line" or self.edit_method == "line2"):
			if event.isAutoRepeat() or self.mode != 'edit':
				return
			self.select_line = True
			self.line2_mode = "select"
			# self.label_seg.MannualMode = True
			print "[Mode - select_line]: ", self.select_line

		elif event.key() == QtCore.Qt.Key_C and self.edit_method == "line":
			if event.isAutoRepeat() or self.mode != 'edit':
				return
			self.cutline_mode = True
			print "[Mode - cutline]: ", self.cutline_mode

		elif event.key() == QtCore.Qt.Key_B and self.edit_method == "polygon":
			if event.isAutoRepeat():
				return
			self.label_seg.disp_segBI_on()

		elif event.key() == QtCore.Qt.Key_N:
			if event.isAutoRepeat():
				return
			self.label_seg.disp_ori_on()

		elif event.key() == QtCore.Qt.Key_M and self.edit_method == "line2":
			if event.isAutoRepeat():
				return
			self.label_seg.disp_finalID_on()

		elif event.key() == QtCore.Qt.Key_Equal and self.edit_method == "line2":
			if event.isAutoRepeat() or self.mode != 'edit':
				return
			self.sp.setValue(self.sp.value() + 1)

		elif event.key() == QtCore.Qt.Key_Minus and self.edit_method == "line2":
			if event.isAutoRepeat() or self.mode != 'edit':
				return
			self.sp.setValue(self.sp.value() - 1)

		elif event.key() == QtCore.Qt.Key_R and self.edit_method == "line":
			if event.isAutoRepeat() or self.mode != 'edit':
				return
			self.label_seg.grabcut_on_bbox()
			print "Run GrabCut on bbox: ", self.label_seg.bbox

		elif event.key() == QtCore.Qt.Key_T:
			## change between tab_method
			if event.isAutoRepeat():
				return
			curIdx = self.tab_widget_method.currentIndex()
			numTabs = self.tab_widget_method.count()
			newIdx = (curIdx + 1) % numTabs
			print ">>>>>> curIdx: {}; numTabls: {}; newIdx: {};".format(curIdx, numTabs, newIdx)
			self.tab_method_onchange(newIdx)

		elif event.key() == QtCore.Qt.Key_A:
			if self.mode == 'edit':
				successful = self.handle_save_button()  # auto save
			else:
				logging.warn('Can not save in view mode!')
			if successful == True:
				# jump to prev; stay in the 1st one if already in the 1st one;
				self.process_idx -= 1
				if self.process_idx == 0:
					logging.warn(u'这是第一张。')
				elif self.process_idx < 0:
					self.process_idx += len(self.images_path)
				# self.process_idx = max(0, self.process_idx - 1)
				self.process_anno(False)
			else:
				logging.warn('Cannot save -- conflict exists in image annotation\n')	

		elif event.key() == QtCore.Qt.Key_D:
			if self.mode == 'edit':
				successful = self.handle_save_button()
			else:
				logging.warn('Can not save in view mode!')
			if successful == True:
				self.process_idx = self.process_idx + 1      # jump to next
				if self.process_idx < len(self.images_path):
					if self.process_idx == len(self.images_path)-1:
						logging.warn(u'已经到达最后一张。')
				else:
					self.process_idx -= len(self.images_path)
				self.process_anno(True)
			else:
				logging.warn('Cannot save -- conflict exists in image annotation\n')	

		elif event.key() == QtCore.Qt.Key_S:
			self.handle_save_button()

	def event(self, event):
		if (event.type() == QtCore.QEvent.KeyPress):
			pass
		elif (event.type() == QtCore.QEvent.MouseButtonRelease):
			if self.mode == 'edit':
				if self.edit_method == 'segmentation':
					# annotate when release mouse button.
					self.label_seg.annotate_superpixel()

		elif (event.type() == QtCore.QEvent.MouseButtonPress):
			if self.edit_method == 'segmentation':
				# start collecting points for current label when
				# mouseButtonPress
				self.label_seg.collect_points = []
		return QtGui.QWidget.event(self, event)

	# def mouseDoubleClickEvent(self, event):
	# 	# print "DoubleClick!!!"

	# 	if self.mode == 'edit' and event.buttons() == QtCore.Qt.LeftButton:
	# 		# automatically connect last point with 1st point and annotate
	# 		# polygon when double click.
	# 		if self.edit_method == 'polygon':
	# 			self.label_seg.update_polygon_value()

	# 			# TEMP! finish marking one line when double click
	# 		if self.edit_method == 'line':
	# 			self.label_seg.update_line_value()

	def on_context_menu(self, event):
		if self.edit_method != "line2":
			return

		self.r_menu.popup(QtGui.QCursor.pos())

	def smartGCutAction_slot(self):
		self.line2_mode = "smartGC"
		print "[{}] ON".format(self.line2_mode)

	def polygonOtsuAction_slot(self):
		self.line2_mode = "polygonOtsu"
		print "[{}] ON".format(self.line2_mode)

	def polygonMinusAction_slot(self):
		self.line2_mode = "polygonMinus"
		print "[{}] ON".format(self.line2_mode)


	def polygonPlusAction_slot(self):
		self.line2_mode = "polygonPlus"
		print "[{}] ON".format(self.line2_mode)


	def undoSegAction_slot(self):
		print "[[Undo Seg]]"
		self.label_seg.undo_seg()

	def undoPointAction_slot(self):
		print "[[Pop Point]]"
		self.label_seg.pop_collect_poly_points()