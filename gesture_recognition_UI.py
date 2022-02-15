import sys
from initUI import Ui_init
from addDataset import Ui_addDataset
from progressbar import Ui_progress
from MainUI import Ui_MainUI
from ModifyUI import Ui_ModifyUI

from PyQt5 import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# 초기 선택 개수 설정
NumOfGesture = 2

class Gesture_recognition(QMainWindow, Ui_MainUI):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()

app = QApplication([])
ui = Gesture_recognition()
QApplication.processEvents()
sys.exit(app.exec_())