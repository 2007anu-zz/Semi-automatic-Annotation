# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'seganno_homeWindow.ui'
#
# Created: Fri Jun 30 17:56:28 2017
#      by: PyQt4 UI code generator 4.11
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_HomeWindow(object):
    def setupUi(self, HomeWindow):
        HomeWindow.setObjectName(_fromUtf8("HomeWindow"))
        HomeWindow.resize(578, 399)
        self.centralwidget = QtGui.QWidget(HomeWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label_SelectJob = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_SelectJob.sizePolicy().hasHeightForWidth())
        self.label_SelectJob.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial Black"))
        font.setPointSize(22)
        font.setBold(False)
        font.setWeight(50)
        self.label_SelectJob.setFont(font)
        self.label_SelectJob.setAlignment(QtCore.Qt.AlignCenter)
        self.label_SelectJob.setObjectName(_fromUtf8("label_SelectJob"))
        self.verticalLayout_2.addWidget(self.label_SelectJob)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))

        self.btn_Parsing = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_Parsing.sizePolicy().hasHeightForWidth())
        self.btn_Parsing.setSizePolicy(sizePolicy)
        self.btn_Parsing.setMinimumSize(QtCore.QSize(40, 18))
        self.btn_Parsing.setObjectName(_fromUtf8("btn_Parsing"))
        self.verticalLayout.addWidget(self.btn_Parsing)

        self.btn_LaneLine = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_LaneLine.sizePolicy().hasHeightForWidth())
        self.btn_LaneLine.setSizePolicy(sizePolicy)
        self.btn_LaneLine.setObjectName(_fromUtf8("btn_LaneLine"))
        self.verticalLayout.addWidget(self.btn_LaneLine)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        HomeWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(HomeWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 578, 23))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        HomeWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(HomeWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        HomeWindow.setStatusBar(self.statusbar)

        self.retranslateUi(HomeWindow)
        QtCore.QMetaObject.connectSlotsByName(HomeWindow)

    def retranslateUi(self, HomeWindow):
        HomeWindow.setWindowTitle(_translate("HomeWindow", "segannoAnnotatingTool", None))
        self.label_SelectJob.setText(_translate("HomeWindow", "Please Select An Annotating Job", None))
        self.btn_Parsing.setText(_translate("HomeWindow", "Parsing ", None))
        self.btn_LaneLine.setText(_translate("HomeWindow", "LaneLine 车道线", None))