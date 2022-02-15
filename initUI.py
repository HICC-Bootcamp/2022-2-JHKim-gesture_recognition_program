# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'initUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_init(object):
    def setupUi(self, init):
        init.setObjectName("init")
        init.resize(1600, 900)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(init.sizePolicy().hasHeightForWidth())
        init.setSizePolicy(sizePolicy)
        init.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(init)
        self.centralwidget.setObjectName("centralwidget")
        self.welcomeMessage = QtWidgets.QLabel(self.centralwidget)
        self.welcomeMessage.setGeometry(QtCore.QRect(500, 110, 661, 431))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(26)
        self.welcomeMessage.setFont(font)
        self.welcomeMessage.setObjectName("welcomeMessage")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(250, 570, 1091, 251))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.buttonLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.buttonLayout.setContentsMargins(100, 0, 100, 0)
        self.buttonLayout.setHorizontalSpacing(100)
        self.buttonLayout.setVerticalSpacing(6)
        self.buttonLayout.setObjectName("buttonLayout")
        self.SelectNumOfGesture = QtWidgets.QComboBox(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SelectNumOfGesture.sizePolicy().hasHeightForWidth())
        self.SelectNumOfGesture.setSizePolicy(sizePolicy)
        self.SelectNumOfGesture.setMinimumSize(QtCore.QSize(396, 110))
        self.SelectNumOfGesture.setMaximumSize(QtCore.QSize(400, 110))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(20)
        font.setKerning(True)
        self.SelectNumOfGesture.setFont(font)
        self.SelectNumOfGesture.setTabletTracking(False)
        self.SelectNumOfGesture.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.SelectNumOfGesture.setObjectName("SelectNumOfGesture")
        self.SelectNumOfGesture.addItem("")
        self.SelectNumOfGesture.addItem("")
        self.SelectNumOfGesture.addItem("")
        self.SelectNumOfGesture.addItem("")
        self.SelectNumOfGesture.addItem("")
        self.SelectNumOfGesture.addItem("")
        self.SelectNumOfGesture.addItem("")
        self.buttonLayout.addWidget(self.SelectNumOfGesture, 0, 0, 1, 1)
        self.okayButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.okayButton.setMaximumSize(QtCore.QSize(200, 110))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(20)
        self.okayButton.setFont(font)
        self.okayButton.setObjectName("okayButton")
        self.buttonLayout.addWidget(self.okayButton, 0, 1, 1, 1)
        init.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(init)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 21))
        self.menubar.setObjectName("menubar")
        init.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(init)
        self.statusbar.setObjectName("statusbar")
        init.setStatusBar(self.statusbar)

        self.retranslateUi(init)
        self.SelectNumOfGesture.currentIndexChanged['int'].connect(init.InitNumOfGesture) # type: ignore
        self.okayButton.clicked.connect(init.okayButtonClicked) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(init)

    def retranslateUi(self, init):
        _translate = QtCore.QCoreApplication.translate
        init.setWindowTitle(_translate("init", "Gesture_recognition"))
        self.welcomeMessage.setText(_translate("init", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">제스처 인식 프로그램을 </span></p><p align=\"center\"><span style=\" font-weight:600;\">이용해주셔서 감사합니다. </span></p><p align=\"center\"><br/></p><p align=\"center\"><span style=\" font-weight:600;\">사용을 위해 </span></p><p align=\"center\"><span style=\" font-weight:600;\">최소 두 개의 제스처를 추가해주세요!</span></p></body></html>"))
        self.SelectNumOfGesture.setItemText(0, _translate("init", "2"))
        self.SelectNumOfGesture.setItemText(1, _translate("init", "3"))
        self.SelectNumOfGesture.setItemText(2, _translate("init", "4"))
        self.SelectNumOfGesture.setItemText(3, _translate("init", "5"))
        self.SelectNumOfGesture.setItemText(4, _translate("init", "6"))
        self.SelectNumOfGesture.setItemText(5, _translate("init", "7"))
        self.SelectNumOfGesture.setItemText(6, _translate("init", "8"))
        self.okayButton.setText(_translate("init", "확인"))

    def InitNumOfGesture(self):
        pass

    def okayButtonClicked(self):
        pass


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    init = QtWidgets.QMainWindow()
    ui = Ui_init()
    ui.setupUi(init)
    init.show()
    sys.exit(app.exec_())
