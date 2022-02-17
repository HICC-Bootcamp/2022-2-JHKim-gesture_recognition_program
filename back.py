dataset_name=[] #dataset 이름
dataset_function=[] #dataset 기능
dataset_image=[] #영상 이름


def datasetIsEmpty(): #dataset이 비어있는지 확인하는 함수, 비어있으면 1 아니면 0을 return
    import os
    path='./dataset'
    length=len(os.listdir(path))
    if (length>0):
        print('dataset is not empty')
        return 0
    else:
        print('dataset is empty')
        return 1

def addInformation(name, function): #이름과 기능을 리스트에 저장하는 함수
    dataset_name.append(name)
    dataset_function.append(function)
    print('dataset_name =',dataset_name)
    print('dataset_function =',dataset_function)

def confirmRepetition():
    n=len(dataset_function)
    for x in range(0,n):
        first=dataset_function[x]
        for y in range(x+1,n):
            second=dataset_function[y]
            if first==second:
                return 1
    return 0

def doFuction(function): #function 동작 (바탕화면, 특정화면캡쳐, 작업관리자, 가상화면생성, 가상화면닫기, 음량up/down/mute)
    import pyautogui
    
    if function=='바탕화면':
        pyautogui.keyDown('win')
        pyautogui.press('d')
        pyautogui.keyUp('win')
    elif function=='특정화면캡쳐':
        pyautogui.keyDown('win')
        pyautogui.keyDown('shift')
        pyautogui.press('s')
        pyautogui.keyUp('win')
        pyautogui.keyUp('shift')
    elif function=='작업관리자':
        pyautogui.keyDown('ctrl')
        pyautogui.keyDown('shift')
        pyautogui.press('esc')
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')
    elif function=='볼륨up':
        from pynput.keyboard import Key, Controller
        keyboard = Controller()
        for i in range(5):
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)
    elif function=='볼륨down':
        from pynput.keyboard import Key, Controller
        keyboard = Controller()
        for i in range(5):
            keyboard.press(Key.media_volume_down)
            keyboard.release(Key.media_volume_down)
    elif function=='음소거':
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        if volume.GetMute():
            volume.SetMute(0, None)
        else:
            volume.SetMute(1, None)

    '''elif function=='가상화면생성': #//issue//새로 생성된 가상 데스크톱에서 프로그램을 별도로 실행시켜야 프로그램이 작동한다.
        pyautogui.keyDown('win')
        pyautogui.keyDown('ctrl')
        pyautogui.press('d')
        pyautogui.keyUp('win')
        pyautogui.keyUp('ctrl')
    elif function=='가상화면닫기':
        pyautogui.keyDown('win')
        pyautogui.keyDown('ctrl')
        pyautogui.press('F4')
        pyautogui.keyUp('win')
        pyautogui.keyUp('ctrl')'''

def countNumOfDataset():
    import os
    path = './dataset'
    length = len(os.listdir(path))
    if (length >= 2):
        print('More than 2 datasets exist')
        return 1
    else:
        print('Less than 2 datasets exist')
        return 0

def deleteGesture(name,function,dataset):
    dataset_name.remove(name)
    dataset_function.remove(function)
    dataset_image.remove(dataset)

def getDataset_name():
    return dataset_name

def getDataset_function():
    return dataset_function

def getDataset_image():
    return dataset_image

def RecordGesture():


def trainModel():


#백에서 사용 예정인 함수
def addDataset_image(image):
    dataset_name.append(image)
















