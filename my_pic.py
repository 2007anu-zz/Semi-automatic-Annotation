#!/usr/bin/python
# -*- coding: utf-8 -*-

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


class MyPic(QtGui.QLabel):
	def __init__(self, *args):
		self.img_arr = None
		self.main_window = None
		QtGui.QLabel.__init__(self, *args)
		self.disp_size = None   # (W, H)\

	def simple_draw_circle(self, center, r=10, colorFill=(0, 0, 255), colorLine=(0, 0, 255)):
		# center (y, x)
		# draw on self.img_arr
		cy, cx = center
		img = Image.fromarray(self.img_arr)
		draw = ImageDraw.Draw(img)
		draw.ellipse((cx - r, cy - r, cx + r, cy + r),
					 fill=colorFill, outline=colorLine)
		self.img_arr = np.array(img)

	def event(self, event):
		et = event.type()
		if (et == QtCore.QEvent.MouseButtonPress):
			pass
		elif (et == QtCore.QEvent.KeyPress):
			print event.key(), '>>>>>key pressed'
		return QtGui.QLabel.event(self, event)

	def read_image(self, img_path):
		if img_path is None:
			self.img_arr = np.zeros((720, 1280, 3), dtype=np.uint8)
			# self.img_arr = np.zeros((10,10,3), dtype=np.uint8) #
			# self.img_arr = np.zeros((720,960,3), dtype=np.uint8)
		else:
			self.img_arr = read_image_arr(img_path)

	def update(self):
		if self.disp_size:
			tmp_img = np.array(
				Image.fromarray(
					self.img_arr).resize(
					self.disp_size))
			self.setPixmap(QtGui.QPixmap(cvtrgb2qtimage(tmp_img)))
			print self.disp_size, 'hahah'
		else:
			self.setPixmap(QtGui.QPixmap(cvtrgb2qtimage(self.img_arr)))

	def set_img(self, img):
		self.img_arr = img

	def set_main_window(self, main_window_obj):
		self.main_window = main_window_obj