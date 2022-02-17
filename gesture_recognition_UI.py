import sys
from PyQt5 import uic
from PyQt5.uic import loadUi

from PyQt5 import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

NumofGesture = 2

class Gesture_recognition(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/initUI.ui", self)

        self.SelectNumOfGesture.currentIndexChanged['int'].connect(self.InitNumOfGesture)
        self.okayButton.clicked.connect(self.okayButtonClicked)

    @pyqtSlot()
    def InitNumOfGesture(self):
        global NumofGesture
        NumofGesture = int(self.SelectNumOfGesture.currentText())
        print(NumofGesture)


    def okayButtonClicked(self):
        adddata = addDataset_UI()
        widget.addWidget(adddata)
        widget.setCurrentIndex(widget.currentIndex() + 1)


class addDataset_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/addDataset.ui", self)

        self.current = 0
        self.okayButton2.clicked.connect(self.okayButton2Clicked)

    def okayButton2Clicked(self):
        if(self.current >= NumofGesture):
            progress = progressbar_UI()
            widget.addWidget(progress)
            widget.setCurrentIndex(widget.currentIndex() + 1)
        else:
            self.current += 1
            name = self.lineEdit_name.text()
            function = self.comboBox_function.currentText()
            print(name)
            print(function)


class progressbar_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/progressbar.ui", self)

        self.okayButton.setEnabled(False)

        self.timerVar = QTimer()
        self.timerVar.setInterval(1000)
        self.timerVar.timeout.connect(self.progressBarTimer)
        self.timerVar.start()

        self.okayButton.clicked.connect(self.okayButtonClicked)

    def progressBarTimer(self):
        self.time = self.progressBar.value()
        self.time += 10
        self.progressBar.setValue(self.time)

        if self.time >= self.progressBar.maximum():
            self.timerVar.stop()
            self.okayButton.setEnabled(True)

    def okayButtonClicked(self):
        mainui = Main_UI()
        widget.addWidget(mainui)
        widget.setCurrentIndex(widget.currentIndex() + 1)


class Main_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/MainUI.ui", self)

        self.start_button.clicked.connect(self.startButtonClicked)
        self.edit_button.clicked.connect(self.editButtonClicked)

    def startButtonClicked(self):
        pass

    def editButtonClicked(self):
        modifyui = ModifyUI()
        widget.addWidget(modifyui)
        widget.setCurrentIndex(widget.currentIndex() + 1)


class ModifyUI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/ModifyUI.ui")

        # 남은 버튼들 만들어서 정리하면 된다.


if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = QStackedWidget()
    initui = Gesture_recognition()

    widget.setWindowTitle("Gesture_recognition")
    widget.addWidget(initui)

    widget.setFixedWidth(1600)
    widget.setFixedHeight(900)
    widget.show()

    sys.exit(app.exec_())
