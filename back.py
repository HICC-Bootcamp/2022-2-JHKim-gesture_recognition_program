dataset_name=[] #dataset 이름
dataset_function=[] #dataset 기능

def datasetIsEmpty(): #dataset이 비어있는지 확인하는 함수, 비어있으면 1 아니면 0을 return
    import os
    path='./dataset'
    length=len(os.listdir(path))
    if(length>0):
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


