# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'progressbar.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_progress(object):
    def setupUi(self, progress):
        progress.setObjectName("progress")
        progress.resize(1600, 900)
        progress.setAnimated(False)
        self.centralwidget = QtWidgets.QWidget(progress)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(610, 120, 421, 271))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(26)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(350, 440, 971, 161))
        self.progressBar.setProperty("value", 66)
        self.progressBar.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.progressBar.setTextVisible(True)
        self.progressBar.setOrientation(QtCore.Qt.Horizontal)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.progressBar.setObjectName("progressBar")
        self.okayButton = QtWidgets.QPushButton(self.centralwidget)
        self.okayButton.setGeometry(QtCore.QRect(1230, 700, 91, 81))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(20)
        self.okayButton.setFont(font)
        self.okayButton.setCheckable(True)
        self.okayButton.setObjectName("okayButton")
        progress.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(progress)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 21))
        self.menubar.setObjectName("menubar")
        progress.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(progress)
        self.statusbar.setObjectName("statusbar")
        progress.setStatusBar(self.statusbar)

        self.retranslateUi(progress)
        QtCore.QMetaObject.connectSlotsByName(progress)

    def retranslateUi(self, progress):
        _translate = QtCore.QCoreApplication.translate
        progress.setWindowTitle(_translate("progress", "Gesture_recognition"))
        self.label.setText(_translate("progress", "제스처를 학습중입니다.\n"
"\n"
" 잠시만 기다려주세요."))
        self.okayButton.setText(_translate("progress", "확인"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    progress = QtWidgets.QMainWindow()
    ui = Ui_progress()
    ui.setupUi(progress)
    progress.show()
    sys.exit(app.exec_())
