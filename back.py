dataset_name=[] #dataset 이름
dataset_function=[] #dataset 기능

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

def ConfirmRepetition():
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













