import sys
from PyQt5.uic import loadUi
from create_dataset import startVideoCapture

from back import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

#초기에 생성하는 제스처의 수
NumofGesture = 2

#나중에 추가할 때 저장하고자 하는 곳을 기억하는 변수
add_id = 0

#Gesture_recognition: 맨 처음 화면
class Gesture_recognition(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/welcomeUI.ui", self)

        self.startButton.clicked.connect(self.startButtonClicked)

    def startButtonClicked(self):
        # 데이터 셋이 비어있으면 초기화면으로, 아니면 메인화면으로 간다.
        if(not datasetIsEmpty()):
            widget.setCurrentIndex(widget.currentIndex() + 4)
        else:
            widget.setCurrentIndex(widget.currentIndex() + 1)


#initUI: 등록할 제스처의 수를 입력 받는 UI
class initUI(QMainWindow):
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

    def okayButtonClicked(self):
        widget.setCurrentIndex(widget.currentIndex() + 1)


#addDataset_UI: 데이터셋을 추가할 수 있는 UI(이름, 기능, 화면 녹화)
class addDataset_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/addDataset.ui", self)

        self.current = 0    #현재 반복 수
        #확인 버튼을 누르면 이름과 기능을 저장한 후 영상을 녹화한다.
        self.okayButton.clicked.connect(self.okayButtonClicked)

    def okayButtonClicked(self):
        # 현재 반복 횟수가 초기에 설정한 제스처 수와 같아질 때 저장을 멈추고 학습화면으로 넘어간다.
        if(self.current >= NumofGesture):
            widget.setCurrentIndex(widget.currentIndex() + 1)
        else:
            name = self.lineEdit_name.text()
            function = self.comboBox_function.currentText()

            # back.py에 이름과 기능 저장
            addInformation(name, function)

            # 기능이 중복이라면 추가하지 않는다. (오류발생 : 한 개 추가하고 두 번째 추가할 때 전부 중복 처리된다.)
            if(confirmRepetition()):
                QMessageBox.warning(self, "기능 중복 발생", "기능이 중복되었습니다. 다른 기능을 선택하세요.")
                return

            self.current += 1
            startVideoCapture(self.current, name)


#progressbar_UI: 학습 상황을 progressBar를 통해 보여준다.
class progressbar_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/progressbar.ui", self)

        self.progressBar.setStyleSheet("QProgressBar{\n"
                                       "    background-color: rgb(255, 255, 255);\n"
                                       "    color:rgb(0,0,0);\n"
                                       "    border-style: none;\n"
                                       "    border-bottom-right-radius: 10px;\n"
                                       "    border-bottom-left-radius: 10px;\n"
                                       "    border-top-right-radius: 10px;\n"
                                       "    border-top-left-radius: 10px;\n"
                                       "    text-align: center;\n"
                                       "}\n"
                                       "QProgressBar::chunk{\n"
                                       "    border-bottom-right-radius: 10px;\n"
                                       "    border-bottom-left-radius: 10px;\n"
                                       "    border-top-right-radius: 10px;\n"
                                       "    border-top-left-radius: 10px;\n"
                                       "    background-color: rgb(175, 152, 216);\n"
                                       "}\n"
                                       "\n"
                                       "")

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
        widget.setCurrentIndex(widget.currentIndex() + 1)


#MainUI: 초기에 등록을 완료하고 제스처를 실행, 수정할 수 있는 UI
class Main_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/MainUI.ui", self)

        self.start_button.clicked.connect(self.startButtonClicked)
        self.edit_button.clicked.connect(self.editButtonClicked)

    # opencv연동해서 아마도(test.py) 연동해서 실행해야 할 듯
    def startButtonClicked(self):
        pass

    def editButtonClicked(self):
        widget.setCurrentIndex(widget.currentIndex() + 1)


#ModifyUi: 등록된 제스처의 이름, 기능, 영상을 보여주는 화면, (확인, 수정이 가능하다.)
class ModifyUI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/ModifyUI.ui", self)

        self.nameList = [self.name_1, self.name_2, self.name_3, self.name_4, self.name_5, self.name_6, self.name_7, self.name_8]
        self.functionList = [self.comboBox_function_1, self.comboBox_function_2, self.comboBox_function_3, \
                             self.comboBox_function_4, self.comboBox_function_5, self.comboBox_function_6, \
                             self.comboBox_function_7, self.comboBox_function_8]

        self.plus_1.clicked.connect(lambda: self.plusButtonClicked(1))
        self.plus_2.clicked.connect(lambda: self.plusButtonClicked(2))
        self.plus_3.clicked.connect(lambda: self.plusButtonClicked(3))
        self.plus_4.clicked.connect(lambda: self.plusButtonClicked(4))
        self.plus_5.clicked.connect(lambda: self.plusButtonClicked(5))
        self.plus_6.clicked.connect(lambda: self.plusButtonClicked(6))
        self.plus_7.clicked.connect(lambda: self.plusButtonClicked(7))
        self.plus_8.clicked.connect(lambda: self.plusButtonClicked(8))

        self.delete_1.clicked.connect(lambda: self.deleteButtonClicked(1))
        self.delete_2.clicked.connect(lambda: self.deleteButtonClicked(2))
        self.delete_3.clicked.connect(lambda: self.deleteButtonClicked(3))
        self.delete_4.clicked.connect(lambda: self.deleteButtonClicked(4))
        self.delete_5.clicked.connect(lambda: self.deleteButtonClicked(5))
        self.delete_6.clicked.connect(lambda: self.deleteButtonClicked(6))
        self.delete_7.clicked.connect(lambda: self.deleteButtonClicked(7))
        self.delete_8.clicked.connect(lambda: self.deleteButtonClicked(8))

        self.save_1.clicked.connect(lambda: self.saveButtonClicked(1))
        self.save_2.clicked.connect(lambda: self.saveButtonClicked(2))
        self.save_3.clicked.connect(lambda: self.saveButtonClicked(3))
        self.save_4.clicked.connect(lambda: self.saveButtonClicked(4))
        self.save_5.clicked.connect(lambda: self.saveButtonClicked(5))
        self.save_6.clicked.connect(lambda: self.saveButtonClicked(6))
        self.save_7.clicked.connect(lambda: self.saveButtonClicked(7))
        self.save_8.clicked.connect(lambda: self.saveButtonClicked(8))

        self.okayButton.clicked.connect(self.okayButtonClicked)

    def plusButtonClicked(self, id):
        global add_id
        add_id = id - 1
        self.adddata_after = addDataset_after()
        self.adddata_after.show()

    def deleteButtonClicked(self, id):
        name = self.nameList[id-1].text()
        func = self.functionList[id-1].currentText()
        print(name)
        print(func)

    def saveButtonClicked(self, id):
        name = self.nameList[id - 1].text()
        func = self.functionList[id - 1].currentText()
        print(name)
        print(func)

    # 데이터 셋 변동이 있으면 학습화면으로 아니면 메인 화면으로 간다.
    def okayButtonClicked(self):
        if(False):
           widget.setCurrentIndex(widget.currentIndex() - 2)
        else:
            widget.setCurrentIndex(widget.currentIndex() - 1)


class addDataset_after(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/addDataset.ui", self)

        self.okayButton.clicked.connect(self.okayButtonClicked)

    def okayButtonClicked(self):
        name = self.lineEdit_name.text()
        function = self.comboBox_function.currentText()
        print(add_id)
        print(name)
        print(function)
        #영상 녹화 후 팝업 창 종료
        self.close()



if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = QStackedWidget()
    gestureRecognitionProgram = Gesture_recognition()
    initui = initUI()
    adddata = addDataset_UI()
    progress = progressbar_UI()
    mainui = Main_UI()
    modifyui = ModifyUI()

    widget.setWindowTitle("Gesture_recognition")

    widget.addWidget(gestureRecognitionProgram)
    widget.addWidget(initui)
    widget.addWidget(adddata)
    widget.addWidget(progress)
    widget.addWidget(mainui)
    widget.addWidget(modifyui)

    widget.setFixedWidth(1600)
    widget.setFixedHeight(900)
    widget.show()

    sys.exit(app.exec_())