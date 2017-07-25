#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
ver. 2.1.0
Usage:
python main.py
"""
import sys

from PyQt4 import QtGui
from PyQt4.QtGui import *
from PyQt4 import QtCore
from PyQt4.QtCore import Qt

from segimg import SegPic, MyPic
from homeWindow import Ui_HomeWindow
from windowLaneLine import LaneLineWindow
from windowParsing import ParsingWindow

class HomeMenu(QtGui.QMainWindow, Ui_HomeWindow):
	def __init__(self, parent=None):
		super(HomeMenu, self).__init__(parent)
		self.setupUi(self)
		
		self.btn_LaneLine.clicked.connect(self.handle_LaneLine_selected)
		self.btn_Parsing.clicked.connect(self.handle_Parsing_selected)


	def handle_LaneLine_selected(self):
		print "#### <LaneLine Tool> ####"
		winLaneLine = LaneLineWindow()
		winLaneLine.show()
		self.close()


	def handle_Parsing_selected(self):
		print "#### <Parsing Tool> ####"
		winParsing = ParsingWindow()
		winParsing.show()
		self.close()



def main():
	reload(sys)
	sys.setdefaultencoding("utf-8")
	app = QtGui.QApplication(sys.argv)
	# w = MyWindow()
	w = HomeMenu()
	w.show()
	# QtCore.QTimer.singleShot(0, w.appinit)
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()