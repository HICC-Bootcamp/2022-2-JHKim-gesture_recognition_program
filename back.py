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

def deleteDatasetNameFunction(name,func):
    dataset_name.remove(name)
    dataset_function.remove(func)

#def trainModel():


def RecordGesture(idx, name):
    import cv2
    import mediapipe as mp
    import numpy as np
    import time, os

    import sys
    import io
    # 영상 녹화 사전 준비
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

    actions = [name]
    seq_length = 10  # 10으로 변경
    secs_for_action = 10  # 10으로 변경

    # mediapipe initialize
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # webcam initialize
    cap = cv2.VideoCapture(0)

    created_time = int(time.time())

    # make folder for save datasets
    os.makedirs('dataset', exist_ok=True)

    while cap.isOpened():
        for action in enumerate(actions):  # action gesture 마다 녹화
            data = []

            ret, img = cap.read()  # 캠에서 이미지 읽어 와서

            img = cv2.flip(img, 1)  # flip(반전)을 시켜 준다.

            width = int(cap.get(3))  # 폭 웹캠의 성질 그대로
            height = int(cap.get(4))  # 높이 웹캠의 성질 그대로

            fourcc = cv2.VideoWriter_fourcc(*'MP4V')

            # 비디오 저장을 위한 객체를 생성해 줌.
            out = cv2.VideoWriter('%s.mp4' % name, fourcc, 20.0, (width, height))

            # ready time, 녹화 하기 전에 준비할 수 있도록 멘트 주고 3초 대기
            cv2.putText(img, f'Waiting for collecting %s action...' % name, org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            cv2.imshow('img', img)
            cv2.waitKey(3000)

            start_time = time.time()

            while time.time() - start_time < secs_for_action:  # 설정한 시간 만큼 반복
                ret, img = cap.read()  # frame 영상 저장 위해 추가
                ret, frame = cap.read()

                out.write(frame)

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)  # 결과를 mediapipe에 넣어 준다.
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # 결과를 가지고 값들을 뽑아 내는 과정
                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # Compute angles between joints
                        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                             :3]  # Parent joint
                        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                             :3]  # Child joint
                        v = v2 - v1  # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                    :]))  # [15,]

                        angle = np.degrees(angle)  # Convert radian to degree

                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, idx)  # 0, 1, 2

                        d = np.concatenate([joint.flatten(), angle_label])

                        data.append(d)

                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)  # 랜드마크 그린다.

                cv2.imshow('img', img)
                #if cv2.waitKey(1) == ord('q'):
                #   break
            cv2.destroyAllWindows()

            data = np.array(data)  # 데이터를 모았으면 numpy 배열로 변환
            print(action, data.shape)
            np.save(os.path.join('dataset', f'raw_%d, %s_{created_time}' % (idx, name)), data)  # npy형식으로 파일 저장

            # Create sequence data
            full_seq_data = []
            for seq in range(len(data) - seq_length):
                full_seq_data.append(data[seq:seq + seq_length])

            full_seq_data = np.array(full_seq_data)
            print(action, full_seq_data.shape)
            np.save(os.path.join('dataset', f'seq_%d, %s_{created_time}'% (idx, name)), full_seq_data)
        break


#백에서 사용 예정인
def addDataset_image(image):
    dataset_image.append(image)
    print('dataset_image =', dataset_image)

def changeModel(num, name, func):
    dataset_name[num]=name
    dataset_function[num]=func
    print(name,', ', func,'이 등록되었습니다.')

def ReadDatasetInformation():
    import os
    if os.path.isfile('datasetInformation.txt'):
        print('Dataset 정보 파일 존재')
        f = open("datasetInformation.txt", "r")
        line = f.readline()
        list=line.split(' ')
        list.remove('\n')
        dataset_name=list
        print('dataset_name = ',dataset_name)

        line = f.readline()
        list = line.split(' ')
        list.remove('\n')
        dataset_function = list
        print('dataset_function = ', dataset_function)

        line = f.readline()
        dataset_image = line.split(' ')
        dataset_image.remove('')
        print('dataset_image = ', dataset_image)

        f.close()
    else:
        print("Dataset 정보 파일 존재X")

def WriteDatasetInformation():
    f = open("datasetInformation.txt", "w")
    for n in range(0, len(dataset_name)):
        f.write(dataset_name[n])
        f.write(' ')
    f.write('\n')
    for n in range(0, len(dataset_name)):
        f.write(dataset_function[n])
        f.write(' ')
    f.write('\n')
    for n in range(0, len(dataset_name)):
        f.write(dataset_image[n])
        f.write(' ')
    f.close()























