# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 713)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pic = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pic.sizePolicy().hasHeightForWidth())
        self.pic.setSizePolicy(sizePolicy)
        self.pic.setMinimumSize(QtCore.QSize(0, 400))
        self.pic.setText("")
        self.pic.setObjectName("pic")
        self.verticalLayout.addWidget(self.pic)
        self.loadPic = QtWidgets.QPushButton(self.centralwidget)
        self.loadPic.setObjectName("loadPic")
        self.verticalLayout.addWidget(self.loadPic)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setMaximumSize(QtCore.QSize(16777215, 28))
        self.frame.setObjectName("frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout_2.setContentsMargins(-1, -1, -1, 1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.Scale = QtWidgets.QLabel(self.frame)
        self.Scale.setObjectName("Scale")
        self.horizontalLayout_2.addWidget(self.Scale)
        self.Min = QtWidgets.QLabel(self.frame)
        self.Min.setMaximumSize(QtCore.QSize(16777215, 118))
        self.Min.setObjectName("Min")
        self.horizontalLayout_2.addWidget(self.Min)
        self.Feature = QtWidgets.QLabel(self.frame)
        self.Feature.setObjectName("Feature")
        self.horizontalLayout_2.addWidget(self.Feature)
        self.verticalLayout.addWidget(self.frame)
        self.frame1 = QtWidgets.QFrame(self.centralwidget)
        self.frame1.setMaximumSize(QtCore.QSize(16777215, 41))
        self.frame1.setObjectName("frame1")
        self._2 = QtWidgets.QHBoxLayout(self.frame1)
        self._2.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self._2.setContentsMargins(-1, 9, -1, 1)
        self._2.setObjectName("_2")
        self.ScaleFactor = QtWidgets.QDoubleSpinBox(self.frame1)
        self.ScaleFactor.setDecimals(3)
        self.ScaleFactor.setMinimum(1.0)
        self.ScaleFactor.setMaximum(31.0)
        self.ScaleFactor.setSingleStep(0.005)
        self.ScaleFactor.setProperty("value", 1.0)
        self.ScaleFactor.setObjectName("ScaleFactor")
        self._2.addWidget(self.ScaleFactor)
        self.minNeighbours = QtWidgets.QSpinBox(self.frame1)
        self.minNeighbours.setMinimum(1)
        self.minNeighbours.setMaximum(30)
        self.minNeighbours.setObjectName("minNeighbours")
        self._2.addWidget(self.minNeighbours)
        self.Number = QtWidgets.QSpinBox(self.frame1)
        self.Number.setMaximum(2000)
        self.Number.setObjectName("Number")
        self._2.addWidget(self.Number)
        self.verticalLayout.addWidget(self.frame1)
        self.horizontalFrame = QtWidgets.QFrame(self.centralwidget)
        self.horizontalFrame.setMaximumSize(QtCore.QSize(16777215, 27))
        self.horizontalFrame.setObjectName("horizontalFrame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalFrame)
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.custom = QtWidgets.QPushButton(self.horizontalFrame)
        self.custom.setObjectName("custom")
        self.horizontalLayout.addWidget(self.custom)
        self.verticalLayout.addWidget(self.horizontalFrame)
        self.listCascade = QtWidgets.QListWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listCascade.sizePolicy().hasHeightForWidth())
        self.listCascade.setSizePolicy(sizePolicy)
        self.listCascade.setObjectName("listCascade")
        self.verticalLayout.addWidget(self.listCascade)
        self.listMode = QtWidgets.QListWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listMode.sizePolicy().hasHeightForWidth())
        self.listMode.setSizePolicy(sizePolicy)
        self.listMode.setObjectName("listMode")
        item = QtWidgets.QListWidgetItem()
        self.listMode.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listMode.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listMode.addItem(item)
        self.verticalLayout.addWidget(self.listMode)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadPic.setText(_translate("MainWindow", "Загрузить фото"))
        self.Scale.setText(_translate("MainWindow", "                                 ScaleFactor                               "))
        self.Min.setText(_translate("MainWindow", "                               minNeighbours                              "))
        self.Feature.setText(_translate("MainWindow", "                        Number of feature"))
        self.custom.setText(_translate("MainWindow", "Использовать кастомные параметры"))
        __sortingEnabled = self.listMode.isSortingEnabled()
        self.listMode.setSortingEnabled(False)
        item = self.listMode.item(0)
        item.setText(_translate("MainWindow", "Detection"))
        item = self.listMode.item(1)
        item.setText(_translate("MainWindow", "Points"))
        item = self.listMode.item(2)
        item.setText(_translate("MainWindow", "Haar"))
        self.listMode.setSortingEnabled(__sortingEnabled)

