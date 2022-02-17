import sys
from PyQt5 import uic
from PyQt5.uic import loadUi
from PyQt5.QtGui import *

from PyQt5 import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

#초기에 생성하는 제스처의 수
NumofGesture = 2

#initUI: 등록할 제스처의 수를 입력받는 UI
class Gesture_recognition(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/initUI.ui", self)

        #콤보박스로 제스처의 수를 변경하면 그 값이 NumofGesture에 저장된다.
        self.SelectNumOfGesture.currentIndexChanged['int'].connect(self.InitNumOfGesture)

        #확인버튼을 누르면 다음 화면으로 넘어간다.
        self.okayButton.clicked.connect(self.okayButtonClicked)

    def InitNumOfGesture(self):
        global NumofGesture
        NumofGesture = int(self.SelectNumOfGesture.currentText())
        print(NumofGesture)


    def okayButtonClicked(self):
        adddata = addDataset_UI()
        widget.addWidget(adddata)
        widget.setCurrentIndex(widget.currentIndex() + 1)

#addDataset_UI: 데이터셋을 추가할 수 있는 UI(이름, 기능, 화면 녹화)
class addDataset_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/addDataset.ui", self)

        self.current = 0    #현재 반복 수
        #확인 버튼을 누르면 이름과 기능을 저장한 후 영상을 녹화한다.
        self.okayButton2.clicked.connect(self.okayButton2Clicked)

    def okayButton2Clicked(self):
        # 현재 반복 횟수가 초기에 설정한 제스처 수와 같아질 때 저장을 멈추고 학습화면으로 넘어간다.
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


#progressbar_UI: 학습 상황을 progressBar를 통해 보여준다.
class progressbar_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/progressbar.ui", self)

        #100%가 되기 전까지 확인버튼 비활성화
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

    # 확인버튼을 누르면 MainUI로 넘어간다.
    def okayButtonClicked(self):
        mainui = Main_UI()
        widget.addWidget(mainui)
        widget.setCurrentIndex(widget.currentIndex() + 1)


#MainUI: 초기에 등록을 완료하고 제스처를 실행, 수정할 수 있는 UI
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


#ModifyUi: 등록된 제스처의 이름, 기능, 영상을 보여주는 화면, (확인, 수정이 가능하다.)
class ModifyUI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/ModifyUI.ui", self)

        self.modifyButtonsGroup = QButtonGroup(self)
        for button in self.modifyButtonsGroup.findChildren(QAbstractButton):
            self.modifyButtonsGroup.button.clicked.connect(self.saveButtonClicked(button))

        self.okayButton.clicked.connect(self.okayButtonClicked)

    def deleteButtonClicked(self, i):
        pass

    def saveButtonClicked(self):
        print(0)

    def okayButtonClicked(self):
        widget.setCurrentIndex(widget.currentIndex() - 1)


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
