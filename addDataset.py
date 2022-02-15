# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'addDataset.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_addDataset(object):
    def setupUi(self, addDataset):
        addDataset.setObjectName("addDataset")
        addDataset.resize(1600, 900)
        self.centralwidget = QtWidgets.QWidget(addDataset)
        self.centralwidget.setObjectName("centralwidget")
        self.video = QtWidgets.QGraphicsView(self.centralwidget)
        self.video.setGeometry(QtCore.QRect(50, 40, 1501, 591))
        self.video.setMouseTracking(False)
        self.video.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.video.setObjectName("video")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(50, 660, 1501, 161))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.comboBox_function = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboBox_function.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
        self.comboBox_function.setFont(font)
        self.comboBox_function.setObjectName("comboBox_function")
        self.comboBox_function.addItem("")
        self.comboBox_function.addItem("")
        self.comboBox_function.addItem("")
        self.comboBox_function.addItem("")
        self.comboBox_function.addItem("")
        self.comboBox_function.addItem("")
        self.comboBox_function.addItem("")
        self.comboBox_function.addItem("")
        self.gridLayout.addWidget(self.comboBox_function, 0, 1, 1, 1)
        self.lineEdit_name = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_name.setMaximumSize(QtCore.QSize(300, 700))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
        self.lineEdit_name.setFont(font)
        self.lineEdit_name.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lineEdit_name.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.lineEdit_name.setCursorPosition(2)
        self.lineEdit_name.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_name.setObjectName("lineEdit_name")
        self.gridLayout.addWidget(self.lineEdit_name, 0, 0, 1, 1)
        self.okayButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.okayButton.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(20)
        self.okayButton.setFont(font)
        self.okayButton.setObjectName("okayButton")
        self.gridLayout.addWidget(self.okayButton, 0, 2, 1, 1)
        addDataset.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(addDataset)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 21))
        self.menubar.setObjectName("menubar")
        addDataset.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(addDataset)
        self.statusbar.setObjectName("statusbar")
        addDataset.setStatusBar(self.statusbar)

        self.retranslateUi(addDataset)
        QtCore.QMetaObject.connectSlotsByName(addDataset)

    def retranslateUi(self, addDataset):
        _translate = QtCore.QCoreApplication.translate
        addDataset.setWindowTitle(_translate("addDataset", "Gesture_recognition"))
        self.comboBox_function.setItemText(0, _translate("addDataset", "바탕화면으로 가기"))
        self.comboBox_function.setItemText(1, _translate("addDataset", "특정화면 캡처"))
        self.comboBox_function.setItemText(2, _translate("addDataset", "작업관리자 실행"))
        self.comboBox_function.setItemText(3, _translate("addDataset", "가상화면 생성"))
        self.comboBox_function.setItemText(4, _translate("addDataset", "가상화면 닫기"))
        self.comboBox_function.setItemText(5, _translate("addDataset", "음량 up"))
        self.comboBox_function.setItemText(6, _translate("addDataset", "음량 down"))
        self.comboBox_function.setItemText(7, _translate("addDataset", "음소거"))
        self.lineEdit_name.setText(_translate("addDataset", "이름"))
        self.okayButton.setText(_translate("addDataset", "확인"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    addDataset = QtWidgets.QMainWindow()
    ui = Ui_addDataset()
    ui.setupUi(addDataset)
    addDataset.show()
    sys.exit(app.exec_())
