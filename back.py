datasetName = []  # dataset 이름
datasetFunction = []  # dataset 기능

startButtonClickNum = 0


def datasetIsEmpty():  # dataset이 비어있는지 확인하는 함수, 비어있으면 1 아니면 0을 return
    import os
    os.makedirs('dataset', exist_ok=True)
    path = './dataset'
    length = len(os.listdir(path))
    if length > 0:
        print('dataset is not empty')
        return 0
    else:
        print('dataset is empty')
        return 1


def addInformation(name, function):  # 이름과 기능을 리스트에 저장하는 함수
    datasetName.append(name)
    datasetFunction.append(function)
    print('dataset_name =', datasetName)
    print('dataset_function =', datasetFunction)


def confirmRepetition():
    n = len(datasetFunction)
    for x in range(0, n):
        first = datasetFunction[x]
        for y in range(x + 1, n):
            second = datasetFunction[y]
            if first == second:
                return 1
    n = len(datasetName)
    for x in range(0, n):
        first = datasetName[x]
        for y in range(x + 1, n):
            second = datasetName[y]
            if first == second:
                return 1
    return 0


def doFunction(function):  # function 동작 ('바탕화면', '특정화면캡쳐','작업관리자', '볼륨up/down','음소거','위로스크롤','아래로스크롤','닫은탭되돌리기')
    import pyautogui

    if function == '바탕화면':
        pyautogui.keyDown('win')
        pyautogui.press('d')
        pyautogui.keyUp('win')
    elif function == '특정화면캡쳐':
        pyautogui.keyDown('win')
        pyautogui.keyDown('shift')
        pyautogui.press('s')
        pyautogui.keyUp('win')
        pyautogui.keyUp('shift')
    elif function == '작업관리자':
        pyautogui.keyDown('ctrl')
        pyautogui.keyDown('shift')
        pyautogui.press('esc')
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')
    elif function == '볼륨up':
        from pynput.keyboard import Key, Controller
        keyboard = Controller()
        for i in range(5):
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)
    elif function == '볼륨down':
        from pynput.keyboard import Key, Controller
        keyboard = Controller()
        for i in range(5):
            keyboard.press(Key.media_volume_down)
            keyboard.release(Key.media_volume_down)
    elif function == '음소거':
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
    elif function == '위로스크롤':
        import pyautogui
        pyautogui.scroll(500)
    elif function == '아래로스크롤':
        import pyautogui
        pyautogui.scroll(-500)
    elif function == '닫은탭되돌리기':
        import pyautogui
        pyautogui.keyDown('ctrl')
        pyautogui.keyDown('shift')
        pyautogui.press('t')
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')
    else:
        pass

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
    length = len(os.listdir(path)) / 2
    if length > 2:
        print('More than 2 datasets exist')
        return 1
    else:
        print('Less than 2 datasets exist')
        return 0


def getDatasetName():
    return datasetName


def getDatasetFunction():
    return datasetFunction


def deleteDatasetNameFunction(name, func, rep):
    datasetName.remove(name)
    datasetFunction.remove(func)
    if not rep:
        import os
        fileList = os.listdir('./dataset')
        for i in range(0, len(fileList)):
            if name in fileList[i]:
                os.remove('./dataset/' + fileList[i])


def recordGesture(idx, name, func):
    import cv2
    import mediapipe as mp
    import numpy as np
    import time
    import os

    import sys
    import io
    # 영상 녹화 사전 준비
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

    actions = [func]
    seqLength = 10  # 10으로 변경
    secsForAction = 30  # 10으로 변경

    # mediapipe initialize
    mpHands = mp.solutions.hands
    mpDrawing = mp.solutions.drawing_utils
    hands = mpHands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # webcam initialize
    cap = cv2.VideoCapture(0)

    createdTime = int(time.time())

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

            startTime = time.time()

            while time.time() - startTime < secsForAction:  # 설정한 시간 만큼 반복
                ret, img = cap.read()  # frame 영상 저장 위해 추가
                ret, frame = cap.read()

                out.write(frame)

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)  # 결과를 mediapipe 에 넣어 준다.
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

                        angleLabel = np.array([angle], dtype=np.float32)
                        angleLabel = np.append(angleLabel, idx)  # 0, 1, 2

                        d = np.concatenate([joint.flatten(), angleLabel])

                        data.append(d)

                        mpDrawing.draw_landmarks(img, res, mpHands.HAND_CONNECTIONS)  # 랜드마크 그린다.

                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    break

            data = np.array(data)  # 데이터를 모았으면 numpy 배열로 변환
            print(action, data.shape)
            np.save(os.path.join('dataset', f'raw_%d_%s_{createdTime}' % (idx, name)), data)  # npy형식으로 파일 저장

            # Create sequence data
            fullSeqData = []
            for seq in range(len(data) - seqLength):
                fullSeqData.append(data[seq:seq + seqLength])

            fullSeqData = np.array(fullSeqData)
            print(action, fullSeqData.shape)
            np.save(os.path.join('dataset', f'seq_%d_%s_{createdTime}' % (idx, name)), fullSeqData)
        break

    cv2.destroyAllWindows()


def changeModel(num, oldName, name, func):
    global datasetName
    global datasetFunction
    datasetName[num] = name
    datasetFunction[num] = func
    import os
    fileList = os.listdir('./dataset')
    for i in range(0, len(fileList)):
        if oldName in fileList[i]:
            fileName = fileList[i]
            fileOldName = os.path.join('./dataset', fileName)
            fileName = fileName.split('_')
            fileName[2] = name
            fileName = '_'.join(fileName)
            print(fileName)
            fileNewNameNewFile = os.path.join('./dataset', fileName)
            os.rename(fileOldName, fileNewNameNewFile)
            print(fileName)
    print(name, ', ', func, '이 등록되었습니다.')


def changeNameFunction(num, name, func):
    datasetName[num] = name
    datasetFunction[num] = func
    print(datasetName[num], ', ', datasetFunction[num], '이 등록되었습니다.')


def changeVidName(oldName, NewName):
    import os
    fileList = os.listdir()
    for i in range(len(fileList)):
        if oldName in fileList[i]:
            fileOldName = os.path.join('', fileList[i])
            fileNewNameNewFile = os.path.join("", NewName + '.mp4')
            os.rename(fileOldName, fileNewNameNewFile)
            print(fileOldName, '이 ', fileNewNameNewFile, '로 변경되었습니다.')
            break


def deleteVideo(name):
    import os
    fileList = os.listdir()
    for i in range(len(fileList)):
        if name + '.mp4' in fileList[i]:
            os.remove(fileList[i])
            print(fileList[i], '가 삭제 되었습니다.')
            break


def readDatasetInformation():
    import os
    if os.path.isfile('datasetInformation.txt'):
        print('Dataset 정보 파일 존재')
        f = open("datasetInformation.txt", "r")
        line = f.readline()
        list_ = line.split(' ')
        list_.remove('\n')
        global datasetName
        datasetName = list_
        print('dataset_name = ', datasetName)

        line = f.readline()
        list_ = line.split(' ')
        list_.remove('\n')
        global datasetFunction
        datasetFunction = list_
        print('dataset_function = ', datasetFunction)

        f.close()
    else:
        print("Dataset 정보 파일 존재X")


def writeDatasetInformation():
    f = open("datasetInformation.txt", "w")
    for n in range(0, len(datasetName)):
        f.write(datasetName[n])
        f.write(' ')
    f.write('\n')
    for n in range(0, len(datasetName)):
        f.write(datasetFunction[n])
        f.write(' ')
    f.write('\n')

    f.close()


def gestureRecognition():
    import cv2
    import mediapipe as mp
    import numpy as np
    from tensorflow.keras.models import load_model
    import time

    global datasetFunction
    actions = datasetFunction
    seqLength = 10
    model = load_model('models/model.h5')
    mpHands = mp.solutions.hands
    mpDrawing = mp.solutions.drawing_utils
    hands = mpHands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
    # out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

    seq = []
    actionSeq = []
    # before_action='?'
    thisAction = []
    thisAction.append('?')

    while cap.isOpened():
        if startButtonClickNum == 1:
            cv2.destroyAllWindows()
            return

        ret, img = cap.read()
        img0 = img.copy()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                v = v2 - v1  # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                angle = np.degrees(angle)  # Convert radian to degree

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mpDrawing.draw_landmarks(img, res, mpHands.HAND_CONNECTIONS)

                if len(seq) < seqLength:
                    continue

                inputData = np.expand_dims(np.array(seq[-seqLength:], dtype=np.float32), axis=0)

                yPred = model.predict(inputData).squeeze()

                iPred = int(np.argmax(yPred))
                conf = yPred[iPred]

                if conf < 0.9:
                    continue

                action = actions[iPred]
                actionSeq.append(action)

                if len(actionSeq) < 5:
                    continue

                if actionSeq[-1] == actionSeq[-2] == actionSeq[-3] == actionSeq[-4] == actionSeq[-5]:
                    thisAction.append(action)
                    print(thisAction)
                    if len(thisAction) > 3:
                        if thisAction[-1] == thisAction[-2] == thisAction[-3]:
                            doFunction(thisAction[-1])
                            actionSeq.clear()

                            cv2.putText(img, f'{thisAction[-1].upper()}',
                                        org=(int(res.landmark[0].x * img.shape[1]),
                                             int(res.landmark[0].y * img.shape[0] + 20)),
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                                        thickness=2)
                            time.sleep(0.5)

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break


def trainModel():
    import numpy as np
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    actions = datasetFunction

    # dataset에서 seq만 읽기
    fileList_ = os.listdir('./dataset')
    datasetName = []
    for k in range(0, len(fileList_)):
        if 'seq_' in fileList_[k]:
            datasetName.append('./dataset/' + fileList_[k])
    for p in range(0, len(datasetName)):
        print(datasetName[p])
    result = np.load(datasetName[0])
    for i in range(1, len(datasetName)):
        temp = np.load(datasetName[i])
        result = np.concatenate((result, temp), axis=0)

    data = result

    data.shape

    xData = data[:, :, :-1]
    labels = data[:, 0, -1]

    print(xData.shape)
    print(labels.shape)

    from tensorflow.keras.utils import to_categorical

    yData = to_categorical(labels, num_classes=len(actions))
    yData.shape
    from sklearn.model_selection import train_test_split

    xData = xData.astype(np.float32)
    yData = yData.astype(np.float32)

    xTrain, xVal, yTrain, yVal = train_test_split(xData, yData, test_size=0.1, random_state=2022)

    # print(xTrain.shape, yTrain.shap)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    model = Sequential([
        LSTM(64, activation='relu', input_shape=xTrain.shape[1:3]),
        Dense(32, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

    history = model.fit(
        xTrain,
        yTrain,
        validation_data=(xVal, yVal),
        epochs=50,
        callbacks=[
            ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
        ]
    )
    from sklearn.metrics import multilabel_confusion_matrix
    from tensorflow.keras.models import load_model

    model = load_model('models/model.h5')

    yPred = model.predict(xVal)

    multilabel_confusion_matrix(np.argmax(yVal, axis=1), np.argmax(yPred, axis=1))

    return True


def startGesture():
    global startButtonClickNum
    startButtonClickNum = 0


def stopGesture():
    global startButtonClickNum
    startButtonClickNum = 1


def findMaxSeqNum():
    import os
    fileList = os.listdir('./dataset')
    datasetName = []
    datasetSplit = []
    for k in range(0, len(fileList)):
        if 'seq_' in fileList[k]:
            datasetName.append(fileList[k])

    for n in range(len(datasetName)):
        datasetSplit.append(datasetName[n].split('_'))

    print(datasetSplit)

    datasetNum = []
    for seq, num, name, npy in datasetSplit:
        datasetNum.append(int(num))

    print(datasetNum)

    return max(datasetNum)
