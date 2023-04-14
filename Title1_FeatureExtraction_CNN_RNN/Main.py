
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.decomposition import PCA

main = tkinter.Tk()
main.title("Automatic Facial Expression Recognition using Features Extraction Based on Spatial & Temporal Sequences using CNN & RNN Algorithm") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y
global classifier
global pca
names = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

def getID(name):
    index = 0
    for i in range(len(names)):
        if names[i] == name:
            index = i
            break
    return index        
    

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="model")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    
    
def processDataset():
    text.delete('1.0', END)
    global X, Y
    global pca
    '''
    X = []
    Y = []
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            print(name+" "+root+"/"+directory[j])
            if 'Thumbs.db' not in directory[j]:
                img = cv2.imread(root+"/"+directory[j])
                img = cv2.resize(img, (32,32))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(32,32,3)
                X.append(im2arr)
                Y.append(getID(name))
        
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(Y)

    X = X.astype('float32')
    X = X/255    
    test = X[3]
    test = cv2.resize(test,(400,400))
    cv2.imshow("aa",test)
    cv2.waitKey(0)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    np.save('model/X.txt',X)
    np.save('model/Y.txt',Y)
    '''
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    X = np.reshape(X, (X.shape[0],(X.shape[1]*X.shape[2]*X.shape[3])))
    print(X.shape)
    text.insert(END,"Total number of images found in dataset is : "+str(len(X))+"\n")
    text.insert(END,"Total classes found in dataset is : "+str(names)+"\n")
    text.insert(END,"Total features found in dataset before applying features extraction = "+str(X.shape[1])+"\n")
    pca = PCA(n_components = (28*28*3))
    X = pca.fit_transform(X)
    text.insert(END,"Total features found in dataset after applying features extraction = "+str(X.shape[1])+"\n")
    X = np.reshape(X, (X.shape[0],28,28,3))
    print(X.shape)

def trainCNN():
    global classifier
    text.delete('1.0', END)
    if os.path.exists('model/cnnmodel.json'):
        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/cnnmodel_weights.h5")
        classifier.make_predict_function()   
        print(classifier.summary())
        f = open('model/cnnhistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"CNN Training Model Accuracy = "+str(accuracy))
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = 7, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
        classifier.save_weights('model/cnnmodel_weights.h5')            
        model_json = classifier.to_json()
        with open("model/cnnmodel.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/cnnhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/cnnhistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"CNN Training Model Accuracy = "+str(accuracy))


def trainRNN():
    global X
    text.delete('1.0', END)
    if os.path.exists('model/rnnmodel.json'):
        with open('model/rnnmodel.json', "r") as jsonFile:
           loadedModelJson = jsonFile.read()
           lstm_model = model_from_json(loadedModelJson)

        lstm_model.load_weights("model/rnnmodel_weights.h5")
        lstm_model.make_predict_function()   
        print(lstm_model.summary())
        f = open('model/rnnhistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"RNN Training Model Accuracy = "+str(accuracy))
    else:
        X = np.reshape(X, (X.shape[0],X.shape[1],(X.shape[2]*X.shape[3])))
        print(X.shape)
        print(Y.shape)
        lstm_model = Sequential()
        lstm_model.add(LSTM(100, input_shape=(28, 84), activation='relu'))
        lstm_model.add(Dropout(0.5))
        lstm_model.add(Dense(100, activation='relu'))
        lstm_model.add(Dense(7, activation='softmax'))
        
        lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        hist = lstm_model.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
        lstm_model.save_weights('model/rnnmodel_weights.h5')            
        model_json = lstm_model.to_json()
        with open("model/rnnmodel.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/rnnhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/rnnhistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"RNN Training Model Accuracy = "+str(accuracy))
         

def predict():
    global pca
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    image = cv2.resize(image, (32,32))
    img = []
    img.append(image)
    img = np.asarray(img)
    print(img.shape)
    image = np.reshape(img, (img.shape[0],(img.shape[1]*img.shape[2]*img.shape[3])))
    print(image.shape)
    image = pca.transform(image)
    image = np.reshape(image, (image.shape[0],28,28,3))
    #img = cv2.resize(image, (28,28))
    #im2arr = np.array(img)
    #im2arr = im2arr.reshape(1,28,28,3)
    #img = np.asarray(im2arr)
    img = image.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Facial Expression Recognized as : '+names[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Facial Expression Recognized as : '+names[predict], img)
    cv2.waitKey(0)

def graph():
    f = open('model/cnnhistory.pckl', 'rb')
    cnn_data = pickle.load(f)
    f.close()
    cnn_accuracy = cnn_data['accuracy']

    f = open('model/rnnhistory.pckl', 'rb')
    rnn_data = pickle.load(f)
    f.close()
    rnn_accuracy = rnn_data['accuracy']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy')
    plt.plot(cnn_accuracy, 'ro-', color = 'green')
    plt.plot(rnn_accuracy, 'ro-', color = 'orange')
    plt.legend(['CNN Accuracy', 'RNN Accuracy'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Feature Extraction CNN & RNN Accuracy Comparison Graph')
    plt.show()

def exit():
    main.destroy()

font = ('times', 13, 'bold')
title = Label(main, text='Automatic Facial Expression Recognition using Features Extraction Based on Spatial & Temporal Sequences using CNN & RNN Algorithm')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=480,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Facial Emotion Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

cnnButton = Button(main, text="Train CNN Algorithm", command=trainCNN)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1) 

rnnButton = Button(main, text="Train RNN Algorithm", command=trainRNN)
rnnButton.place(x=50,y=250)
rnnButton.config(font=font1) 

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Facial Expression", command=predict)
predictButton.place(x=50,y=350)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=50,y=400)
exitButton.config(font=font1) 

main.config(bg='OliveDrab2')
main.mainloop()
