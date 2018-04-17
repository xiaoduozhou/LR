import numpy as np
import random

def load_file(text_path,label_path):
    text = []
    label = []
    with open(text_path,'r') as f:
         for data in f.readlines():
             data = data[:-1]
             if data != None:
                 data = [float(x) for x in data.split(" ")]
                 data = [x/sum(data) for x in data]
                 data.append(1)  ##tuo zhan hao de
                 text.append(data)

    with open(label_path,'r') as f1:
        for data in f1.readlines():
            data = data[:-1]
            data = float(data)
            label.append(data)
    return np.array(text),np.array(label).reshape((len(np.array(label)),1))

def change_label(label,num):

    labels = []
    for i in label:
        if i == num:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels).reshape((len(np.array(labels)),1))

def Euclidean_distance(B,B_):
    dist = np.sqrt(np.sum(np.square(B - B_)))
    return dist


def p_1(X, B):

    e = np.exp(np.dot(X,B.T))
    e = np.divide(e ,(1+e))
    return e

def knn(data,n,x):
    T = []
    I = []
    Distant = []
    data_ = []
    for i in range(len(data)):
        distant = Euclidean_distance(x,data[i])
        I.append(i)
        Distant.append(distant)
    T = zip(I,Distant)
    T = sorted(T,key= lambda x:x[1])
    k = 0
    for i,t in T:
        if k < n:
            data_.append(data[i])
            k += 1
    return np.array(data_)

def SMOTE(x,label):
    #type = np.zeros((6,300,x.shape[1]))

    for j in range(1,6):
        #print("---------------------------------")
        class_type = []
        for i in range(len(label)):
            if label[i] == j:
                #type[j,k,:] = x[i]
                t= x[i].tolist()
                class_type.append(t)
        class_type = np.array(class_type)

        while class_type.shape[0] < SMOTE_NUM:
           # print(j,class_type.shape[0])
            random_x = class_type[random.randint(0,len(class_type)-1)]
            type_n = knn(class_type,25,random_x)
            for i in range(15):
                rand_num = random.randint(0,24)
                new_x = random_x + random.uniform(0,1)*(type_n[rand_num]-random_x).reshape((1,11))
                new_x[:-1] = 1
                #print(new_x,new_x.shape,class_type.shape)
                class_type = np.r_[class_type,new_x]
                x = np.r_[x,new_x]
                u = np.array([j]).reshape((1,1))
                #print(label.shape)
                label = np.r_[label,u]
    return x,label


def train(X,Y):
    threshold = 0.5
    m,n = np.shape(X)
    B = np.zeros((1,n))
    alpha = 0.15
    while 1:
        alpha = alpha*0.95
        if alpha < 0.001:
            alpha = 0.001
        B_ = B
        P_1 = p_1(X,B)
        # # print(P_1)
        # #print(Y.shape,P_1.shape)
        # print(Y - P_1)
        # print(X)
        # print(((Y - P_1)*X).sum(axis=0))
        delta = ( (Y - P_1)*X).sum(axis=0).reshape(B.shape)
        #B = B + 0.01*delta
        B = B + alpha* delta
        #print(Euclidean_distance(B, B_))
        if Euclidean_distance(B, B_) < threshold:
          break
    return B

def train_PLUS(X,Y):
    m,n = np.shape(X)
    B = np.zeros((1,n))
    alpha = 0.001
    i = 0
    while i< 800:
        i =i+1
        B_ = B
        P_1 = p_1(X,B)
        delta = ( (Y - P_1)*X).sum(axis=0).reshape(B.shape)
        #B = B + 0.01*delta
        B = B + alpha* delta
        #print(Euclidean_distance(B, B_))

    return B





SMOTE_NUM = 1000
text_path = "assign2_dataset/page_blocks_train_feature.txt"
label_path = "assign2_dataset/page_blocks_train_label.txt"
text,label = load_file(text_path,label_path)


test_text_path = "assign2_dataset/page_blocks_test_feature.txt"
test_label_path = "assign2_dataset/page_blocks_test_label.txt"
test_text,test_label = load_file(test_text_path,test_label_path)

B = np.zeros((6,1,11))

text_pro,label_pro = SMOTE(text,label)
np.savetxt("assign2_dataset/text_pro.txt",text_pro)
np.savetxt("assign2_dataset/label_pro.txt",label_pro)

text_pro = np.loadtxt("assign2_dataset/text_pro.txt")
label_pro = np.loadtxt("assign2_dataset/label_pro.txt")




for i in range(1,6):
    print("++++++++++++++++++")
    label_ =  change_label(label_pro,i)
    B[i,:,:] = train_PLUS(text_pro, label_)



np.savetxt("assign2_dataset/B_PLUS.txt",B.reshape((-1)))

B = np.loadtxt("assign2_dataset/B_PLUS.txt").reshape((6,1,11))
predict = np.zeros((len(test_label),1,6))

for i in range(1,6):
    test_label_ = change_label(test_label, i)
    predict[:, :, i] = p_1(test_text, B[i,:,:])
answer = predict.argmax(axis=2)


answer = np.array(answer).reshape(test_label_.shape)
# print(sum(test_label == 1),sum((answer == test_label)&(answer == 1)),sum(answer == 1) )
# print(sum(test_label == 2),sum((answer == test_label)&(answer == 2)),sum(answer == 2) )
# print(sum(test_label == 3),sum((answer == test_label)&(answer == 3)),sum(answer == 3) )
# print(sum(test_label == 4),sum((answer == test_label)&(answer == 4)),sum(answer == 4) )
# print(sum(test_label == 5),sum((answer == test_label)&(answer == 5)),sum(answer == 5) )
print("type_number","right_answer_number","predict_answer_number")
print(sum(test_label == 1),sum((answer == test_label)&(answer == 1)),sum(answer == 1))
print(sum(test_label == 2),sum((answer == test_label)&(answer == 2)),sum(answer == 2))
print(sum(test_label == 3),sum((answer == test_label)&(answer == 3)),sum(answer == 3))
print(sum(test_label == 4),sum((answer == test_label)&(answer == 4)),sum(answer == 4))
print(sum(test_label == 5),sum((answer == test_label)&(answer == 5)),sum(answer == 5))

print("\nprecision : ")
print((sum((answer == test_label)/len(answer)))[0])
print("---------------------------------")
print((sum((answer == test_label)&(answer == 1))/sum(answer == 1))[0],end = '  ')
print((sum((answer == test_label)&(answer == 2))/sum(answer == 2))[0],end = '  ' )
print((sum((answer == test_label)&(answer == 3))/sum(answer == 3))[0],end = '  ')
print((sum((answer == test_label)&(answer == 4))/sum(answer == 4))[0],end = '  ' )
print((sum((answer == test_label)&(answer == 5))/sum(answer == 5))[0] )

print("\nrecall : ")
print((sum((answer == test_label)/len(test_label_)))[0])
print("----------------------------------")
print((sum((answer == test_label)&(answer == 1))/sum(test_label == 1))[0],end = '  ' )
print((sum((answer == test_label)&(answer == 2))/sum(test_label == 2))[0],end = '  ' )
print((sum((answer == test_label)&(answer == 3))/sum(test_label == 3))[0],end = '  ' )
print((sum((answer == test_label)&(answer == 4))/sum(test_label == 4))[0],end = '  ' )
print((sum((answer == test_label)&(answer == 5))/sum(test_label == 5))[0] )
# a = np.array([1,2,3])
# b = np.array([2,3,4])
# print(Euclidean_distance(a,b))
