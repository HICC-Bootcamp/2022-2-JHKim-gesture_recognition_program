import sys
from PyQt5.uic import loadUi

from back import datasetIsEmpty, addInformation, confirmRepetition, deleteDatasetNameFunction, recordGesture
from back import getDatasetName, getDatasetFunction, changeNameFunction
from back import readDatasetInformation, writeDatasetInformation, gestureRecognition, stopGesture, startGesture
from back import changeModel, trainModel, findMaxSeqNum, changeVidName, deleteVideo, countNumOfDataset

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# 초기에 생성하는 제스처의 수
numOfGesture = 2

# GestureRecognition: 맨 처음 화면


class GestureRecognition(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/welcomeUI.ui", self)

        self.startButton.clicked.connect(self.startButtonClicked)

    def startButtonClicked(self):
        # 데이터 셋이 비어있으면 초기화면으로, 아니면 메인화면으로 간다.
        if not datasetIsEmpty():
            widget.setCurrentIndex(widget.currentIndex() + 4)
        else:
            widget.setCurrentIndex(widget.currentIndex() + 1)


# initUI: 등록할 제스처의 수를 입력 받는 UI
class InitUI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/InitUI.ui", self)

        # 콤보박스로 제스처의 수를 변경하면 그 값이 NumofGesture에 저장된다.
        self.selectNumOfGesture.currentIndexChanged['int'].connect(self.initNumOfGesture)

        # 확인버튼을 누르면 다음 화면으로 넘어간다.
        self.okayButton.clicked.connect(self.okayButtonClicked)

    def initNumOfGesture(self):
        global numOfGesture
        numOfGesture = int(self.selectNumOfGesture.currentText())

    def okayButtonClicked(self):
        widget.setCurrentIndex(widget.currentIndex() + 1)


# AddDatasetUI: 데이터셋을 추가할 수 있는 UI(이름, 기능, 화면 녹화)
class AddDatasetUI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/AddDataset.ui", self)

        self.current = 0  # 현재 반복 수
        # 확인 버튼을 누르면 이름과 기능을 저장한 후 영상을 녹화한다.
        self.okayButton.clicked.connect(self.okayButtonClicked)

    def okayButtonClicked(self):
        # 현재 반복 횟수가 초기에 설정한 제스처 수와 같아질 때 저장을 멈추고 학습화면으로 넘어간다.
        if self.current >= numOfGesture:
            writeDatasetInformation()
            widget.setCurrentIndex(widget.currentIndex() + 1)
        else:
            name = self.lineEditName.text()
            function = self.comboBoxFunction.currentText()

            # back.py에 이름과 기능 저장
            addInformation(name, function)

            # 이름, 기능이 중복이라면 추가하지 않는다
            if confirmRepetition():
                deleteDatasetNameFunction(name, function, True)
                QMessageBox.warning(self, "이름, 기능 중복 발생", "이름 혹은 기능이 중복되었습니다.")
                return

            recordGesture(self.current, name, function)
            self.current += 1


# ProgressBarUI: 학습 상황을 progressBar를 통해 보여준다.
class ProgressBarUI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/ProgressBar.ui", self)

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
        # 눌러야 학습시작
        self.startLearning.clicked.connect(self.startLearningButtonClicked)

        # 100%가 되기 전까지 확인버튼 비활성화
        self.okayButton.setEnabled(False)

        self.okayButton.clicked.connect(self.okayButtonClicked)

    def startLearningButtonClicked(self):
        readDatasetInformation()
        self.startLearning.setEnabled(False)
        flag = trainModel()

        if flag:
            self.progressBar.setValue(100)
            self.okayButton.setEnabled(True)

    # 확인버튼을 누르면 MainUI로 넘어간다.
    def okayButtonClicked(self):
        self.progressBar.setValue(0)
        self.startLearning.setEnabled(True)
        self.okayButton.setEnabled(False)
        widget.setCurrentIndex(widget.currentIndex() + 1)


# MainUI: 초기에 등록을 완료하고 제스처를 실행, 수정할 수 있는 UI
class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/MainUI.ui", self)

        self.running = False

        self.startButton.clicked.connect(self.startButtonClicked)
        self.modifyButton.clicked.connect(self.modifyButtonClicked)

    def startButtonClicked(self):
        if not self.running:
            self.startButton.setText('중지')
            self.running = True
            startGesture()
            readDatasetInformation()
            gestureRecognition()

        else:
            self.startButton.setText('제스처 실행')
            self.running = False
            stopGesture()

    def modifyButtonClicked(self):
        widget.setCurrentIndex(widget.currentIndex() + 1)


# ModifyUI: 등록된 제스처의 이름, 기능, 영상을 보여주는 화면, (확인, 수정이 가능하다.)
class ModifyUI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/ModifyUI.ui", self)

        self.isChange = False

        # 이름 입력하는 lineedit 리스트
        self.nameList = [self.name_1, self.name_2, self.name_3, self.name_4, self.name_5,
                         self.name_6, self.name_7, self.name_8]
        # 기능 입력하는 콤보박스 리스트
        self.functionList = [self.combo_box_function_1, self.combo_box_function_2, self.combo_box_function_3,
                             self.combo_box_function_4, self.combo_box_function_5, self.combo_box_function_6,
                             self.combo_box_function_7, self.combo_box_function_8]
        # 추가 버튼 리스트
        self.plusList = [self.plus_1, self.plus_2, self.plus_3, self.plus_4, self.plus_5,
                         self.plus_6, self.plus_7, self.plus_8, ]

        # 영상 재생버튼 리스트
        self.playList = [self.play_1, self.play_2, self.play_3, self.play_4, self.play_5,
                         self.play_6, self.play_7, self.play_8]

        # 저장버튼 리스트
        self.saveList = [self.save_1, self.save_2, self.save_3, self.save_4, self.save_5,
                         self.save_6, self.save_7, self.save_8]

        self.deleteList = [self.delete_1, self.delete_2, self.delete_3, self.delete_4, self.delete_5,
                           self.delete_6, self.delete_7, self.delete_8]

        self.datasetName = list()
        self.datasetFunc = list()
        self.videoList = list()

        # 로드버튼을 누르면 정보가 갱신된다.
        self.loadButton.clicked.connect(self.loadDataset)

        # 재생버튼을 누르면 영상이 재생된다.
        self.play_1.clicked.connect(lambda: self.play(1))
        self.play_2.clicked.connect(lambda: self.play(2))
        self.play_3.clicked.connect(lambda: self.play(3))
        self.play_4.clicked.connect(lambda: self.play(4))
        self.play_5.clicked.connect(lambda: self.play(5))
        self.play_6.clicked.connect(lambda: self.play(6))
        self.play_7.clicked.connect(lambda: self.play(7))
        self.play_8.clicked.connect(lambda: self.play(8))

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

    def loadDataset(self):
        readDatasetInformation()
        self.datasetName = getDatasetName()
        self.datasetFunc = getDatasetFunction()

        nameIdx = 0
        funcIdx = 0

        for i in range(len(self.datasetName)):
            self.nameList[nameIdx].setText(QCoreApplication.translate("", self.datasetName[i]))
            self.plusList[nameIdx].hide()
            self.playList[nameIdx].show()
            self.saveList[nameIdx].show()
            self.deleteList[nameIdx].show()
            nameIdx += 1

        for j in range(len(self.datasetFunc)):
            self.functionList[funcIdx].setCurrentText(self.datasetFunc[j])
            funcIdx += 1

        leftover = nameIdx

        # 남은 것은 모두 빈칸으로 설정
        for idx in range(leftover, 8):
            self.nameList[nameIdx].setText("")
            self.functionList[funcIdx].setCurrentText('---------')
            self.plusList[nameIdx].show()
            self.playList[nameIdx].hide()
            self.saveList[nameIdx].hide()
            self.deleteList[nameIdx].hide()
            nameIdx += 1
            funcIdx += 1

    def play(self, id_):
        import cv2

        videoName = self.datasetName[id_ - 1]

        # 재생할 동영상 파일
        cap = cv2.VideoCapture(videoName+'.mp4')
        fourcc = cv2.VideoWriter_fourcc(* 'XVID')

        while True:
            ret, imgColor = cap.read()

            if not ret:
                break

            cv2.imshow(videoName, imgColor)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    # 어디에 추가버튼을 눌러도 자동정렬이 되므로 id가 크게 의미는 없다.
    def plusButtonClicked(self, id_):
        self.addDataAfterUI = AddDatasetAfter()
        self.addDataAfterUI.show()
        self.isChange = True

    def deleteButtonClicked(self, id_):
        name = self.nameList[id_ - 1].text()
        func = self.functionList[id_ - 1].currentText()

        if countNumOfDataset():
            deleteDatasetNameFunction(name, func, False)
            deleteVideo(name)
            writeDatasetInformation()
            self.nameList[id_ - 1].setText(QCoreApplication.translate(name, ""))
            self.functionList[id_ - 1].setCurrentText('---------')
            QMessageBox.information(self, '삭제 완료', '%s, %s 제스처가 삭제되었습니다.' % (name, func))
            self.isChange = True
        else:
            QMessageBox.warning(self, "제스처 제거 불가", "최소한 두 개 이상의 제스처가 필요합니다.")

    def saveButtonClicked(self, id_):
        oldName = self.datasetName[id_ - 1]
        oldFunc = self.datasetFunc[id_ - 1]
        name = self.nameList[id_ - 1].text()
        func = self.functionList[id_ - 1].currentText()
        changeNameFunction(id_ - 1, name, func)

        if not confirmRepetition():
            changeModel(id_ - 1, oldName, name, func)
            changeVidName(oldName, name)
            writeDatasetInformation()
            QMessageBox.information(self, '수정 완료', '%s, %s로 수정되었습니다.' % (name, func))
            # 바뀐 내용으로 front의 dataset 최신화
            self.datasetName[id_ - 1] = name
            self.datasetFunc[id_ - 1] = func
            self.isChange = True
        else:
            changeNameFunction(id_ - 1, oldName, oldFunc)
            QMessageBox.warning(self, "이름, 기능 중복 발생", "이름 혹은 기능이 중복되었습니다.")
            self.nameList[id_ - 1].setText(oldName)
            self.functionList[id_ - 1].setCurrentText(oldFunc)

    # 데이터 셋 변동이 있으면 학습화면으로 아니면 메인 화면으로 간다.
    def okayButtonClicked(self):
        if self.isChange:
            self.isChange = False
            widget.setCurrentIndex(widget.currentIndex() - 2)
        else:
            widget.setCurrentIndex(widget.currentIndex() - 1)


class AddDatasetAfter(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/AddDataset.ui", self)

        self.okayButton.clicked.connect(self.okayButtonClicked)

    def okayButtonClicked(self):
        name = self.lineEditName.text()
        function = self.comboBoxFunction.currentText()
        addId = findMaxSeqNum() + 1

        addInformation(name, function)

        # 기능이 중복이라면 추가하지 않는다
        if confirmRepetition():
            deleteDatasetNameFunction(name, function, True)
            QMessageBox.warning(self, "이름, 기능 중복 발생", "이름 혹은 기능이 중복되었습니다.")
            return

        # 데이터셋 지우는 함수를 구현하면 실험해보기
        recordGesture(addId, name, function)
        writeDatasetInformation()
        # 영상 녹화 후 팝업 창 종료
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    widget = QStackedWidget()
    gestureRecognitionProgram = GestureRecognition()
    initUI = InitUI()
    addData = AddDatasetUI()
    progress = ProgressBarUI()
    mainUI = MainUI()
    modifyUI = ModifyUI()

    widget.setWindowTitle("Gesture_recognition")

    widget.addWidget(gestureRecognitionProgram)
    widget.addWidget(initUI)
    widget.addWidget(addData)
    widget.addWidget(progress)
    widget.addWidget(mainUI)
    widget.addWidget(modifyUI)

    widget.setFixedWidth(1600)
    widget.setFixedHeight(900)
    widget.show()

    sys.exit(app.exec_())
