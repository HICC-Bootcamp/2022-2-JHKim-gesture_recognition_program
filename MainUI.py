# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainUI(object):
    def setupUi(self, MainUI):
        MainUI.setObjectName("MainUI")
        MainUI.resize(1600, 900)
        self.centralwidget = QtWidgets.QWidget(MainUI)
        self.centralwidget.setObjectName("centralwidget")
        self.video = QtWidgets.QGraphicsView(self.centralwidget)
        self.video.setGeometry(QtCore.QRect(50, 40, 1501, 591))
        self.video.setMouseTracking(False)
        self.video.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.video.setObjectName("video")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(50, 670, 1501, 121))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.buttonLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonLayout.setObjectName("buttonLayout")
        self.start_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.start_button.setMaximumSize(QtCore.QSize(500, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(20)
        self.start_button.setFont(font)
        self.start_button.setObjectName("start_button")
        self.buttonLayout.addWidget(self.start_button, 0, 0, 1, 1)
        self.edit_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.edit_button.setMaximumSize(QtCore.QSize(500, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(20)
        self.edit_button.setFont(font)
        self.edit_button.setObjectName("edit_button")
        self.buttonLayout.addWidget(self.edit_button, 0, 1, 1, 1)
        MainUI.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainUI)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 21))
        self.menubar.setObjectName("menubar")
        MainUI.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainUI)
        self.statusbar.setObjectName("statusbar")
        MainUI.setStatusBar(self.statusbar)

        self.retranslateUi(MainUI)
        QtCore.QMetaObject.connectSlotsByName(MainUI)

    def retranslateUi(self, MainUI):
        _translate = QtCore.QCoreApplication.translate
        MainUI.setWindowTitle(_translate("MainUI", "Gesture_recognition"))
        self.start_button.setText(_translate("MainUI", "제스처 실행"))
        self.edit_button.setText(_translate("MainUI", "제스처 수정"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainUI = QtWidgets.QMainWindow()
    ui = Ui_MainUI()
    ui.setupUi(MainUI)
    MainUI.show()
    sys.exit(app.exec_())
