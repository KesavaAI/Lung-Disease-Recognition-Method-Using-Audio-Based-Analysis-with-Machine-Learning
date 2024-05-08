from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd

import pickle
import os
import librosa
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential

from keras.models import model_from_json
from sklearn.model_selection import train_test_split

main = Tk()
main.title("Respiratory analysis detection of various lung infections using cough signal")
main.geometry("1300x1200")

labels = ['Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'LRTI', 'Pneumonia', 'URTI']

global filename
global classifier


def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir = ".")
    text.insert(END,filename+" loaded\n\n")

    demographic = pd.read_csv('demographic_info.csv',sep=" ")
    demographic.fillna(0, inplace = True)
    diagnosis = pd.read_csv('patient_diagnosis.csv')
    demographic["PATIENT_ID"] = demographic["PATIENT_ID"].astype(int)
    demographic = demographic.merge(diagnosis, on='PATIENT_ID')
    text.insert(END,str(demographic.head()))

def extractFeatures():
    global X, Y
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Total patients records found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total diseases in dataset : "+str(labels)+"\n\n")

def runCNN():
    global classifier, X, Y
    text.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[49] * 100
        text.insert(END,"CNN Training Model Prediction Accuracy = "+str(accuracy))
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (46, 46, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = 8, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        # Splitting data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=50, shuffle=True, verbose=2)
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[49] * 100
        text.insert(END,"CNN Training Model Prediction Accuracy = "+str(accuracy))

    
def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(acc, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('CNN Epoch Wise Accuracy & Loss Graph')
    plt.show()


def predict():
    global classifier
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testAudio")
    x, sr = librosa.load(filename)
    spectrum = librosa.feature.mfcc(x, sr=sr)
    spectrum = spectrum.ravel()
    features = spectrum[0:6348]
    features = features.reshape(46,46,3)
    features = features.astype('float32')
    features = features/255
    temp = []
    temp.append(features)
    temp = np.asarray(temp)
    predict = classifier.predict(temp)
    predict = np.argmax(predict)
    print(predict)
    text.insert(END,"Uploaded Audio contains ["+labels[predict]+"] Disease\n")

def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='Respiratory analysis detection of various lung infections using cough signal')
title.config(bg='mint cream', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
ff = ('times', 12, 'bold')


uploadButton = Button(main, text="Upload Respiratory Audio Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=ff)

featuresButton = Button(main, text="Extract Features from Audio Dataset", command=extractFeatures)
featuresButton.place(x=350,y=100)
featuresButton.config(font=ff)

cnnButton = Button(main, text="Train CNN Algorithm", command=runCNN)
cnnButton.place(x=670,y=100)
cnnButton.config(font=ff)

graphButton = Button(main, text="CNN Accuracy & Loss Graph", command=graph)
graphButton.place(x=880,y=100)
graphButton.config(font=ff)

predictButton = Button(main, text="Upload Test Audio & Predict Disease", command=predict)
predictButton.place(x=50,y=150)
predictButton.config(font=ff)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=350,y=150)
exitButton.config(font=ff)


font1 = ('times', 13, 'bold')
text=Text(main,height=15,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

main.config(bg='gainsboro')
main.mainloop()
