#importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import librosa
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
le = LabelEncoder()

demographic = pd.read_csv('demographic_info.csv',sep=" ")
demographic.fillna(0, inplace = True)
diagnosis = pd.read_csv('patient_diagnosis.csv')


demographic["PATIENT_ID"] = demographic["PATIENT_ID"].astype(int)
print(demographic.info())
print(diagnosis.info())
print(demographic.head())
print(diagnosis.head())

demographic['SEX'] = pd.Series(le.fit_transform(demographic['SEX'].astype(str)))

demographic = demographic.merge(diagnosis, on='PATIENT_ID')
print(demographic)

labels = ['Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'LRTI', 'Pneumonia', 'URTI']

diagnosis = diagnosis.values

def getLabel(name):
    pname = 'none'
    for i in range(len(diagnosis)):
        if name == diagnosis[i,0]:
            pname = diagnosis[i,1]
            break
    for i in range(len(labels)):
        if labels[i] == pname:
            pname = i
            break
    return pname    
            

X = []
Y = []

dups = []
path = 'Respiratory_Sound_Database'
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        if directory[j].endswith(".wav"):
            arr = directory[j].split("_")
            arr = arr[0].strip()
            if arr not in dups:
                dups.append(arr)
                x, sr = librosa.load(root+"/"+directory[j])
                spectrum = librosa.feature.mfcc(x, sr=sr)
                spectrum = spectrum.ravel()
                features = spectrum[0:6348]
                features = features.reshape(46,46,3)
                X.append(features)
                label = getLabel(int(arr))
                Y.append(label)
                print(str(features.shape)+" "+str(arr)+" "+str(label)+" "+directory[j])

X = np.asarray(X)
Y = np.asarray(Y)
np.save("model/X.txt",X)
np.save("model/Y.txt",Y)
labels = ['Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'LRTI', 'Pneumonia', 'URTI']
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')
X = X.astype('float32')
X = X/255
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

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
    print("Training Model Accuracy = "+str(accuracy))
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
    hist = classifier.fit(X, Y, batch_size=16, epochs=50, shuffle=True, verbose=2)
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
    print("Training Model Accuracy = "+str(accuracy))



x, sr = librosa.load('aa2.wav')
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
print(labels[predict])












