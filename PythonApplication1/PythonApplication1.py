import wave, struct
from numpy import array
from keras import models 
from keras import layers
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
#################################################################################################################

filename1='D:/wav/1CHEL_5.WAV'
filename2='D:/wav/AVTO12.WAV'
filename3='D:/wav/abc.WAV'
f1 = wave.open(filename1, 'r')
f2 = wave.open(filename2, 'r') 
f3 = wave.open(filename3, 'r')
N1=f1.getnframes()
N2=f2.getnframes()
N3=f3.getnframes()
print(N1, N2, N3)
bl_len=1000
n_bl1=(N1//bl_len)-0
n_bl2=(N2//bl_len)-0
n_bl3=(N3//bl_len)-0
n_bl=n_bl1+n_bl2+n_bl3
ed_arr=[[20000] * bl_len for i in range(n_bl)]
ed_lab=[0 for i in range(n_bl)]

for i in range(n_bl1):
    ed_lab[i]=0
    for j in range(bl_len):
        decod = struct.unpack("<hh", f1.readframes(1))
        ed_arr[i][j]=decod[1]
for i in range(n_bl2):
    ii=i+n_bl1
    ed_lab[ii]=1
    for j in range(bl_len):
        decod = struct.unpack("<hh", f2.readframes(1))
        ed_arr[ii][j]=decod[1]
for i in range(n_bl3):
    iii=i+n_bl1+n_bl2
    ed_lab[iii]=2
    for j in range(bl_len):
        decod = struct.unpack("<hh", f3.readframes(1))
        ed_arr[iii][j]=decod[1]
ed_arr_numpy = array( ed_arr )
ed_lab_numpy=array(ed_lab)
print(ed_arr_numpy)
print(ed_arr_numpy.shape)
print(ed_lab_numpy)
print(ed_lab_numpy.shape)
f1.close()
f2.close()
f3.close()
#ed_arr_numpy = ed_arr_numpy.reshape((457, 1000, 1)) # CNN
ed_arr_numpy = ed_arr_numpy.reshape((n_bl, bl_len,1))
ed_arr_numpy = ed_arr_numpy.astype('float32') / 32000

############################################################################################

filename1='D:/wav/1CHEL_6.WAV'
filename2='D:/wav/AVTO1.WAV'
filename3='D:/wav/abc2.WAV'
f1 = wave.open(filename1, 'r')
f2 = wave.open(filename2, 'r')
f3 = wave.open(filename3, 'r')
N1=f1.getnframes()
N2=f2.getnframes()
N3=f3.getnframes()
bl_len=1000
n_bl1=(N1//bl_len)-0
n_bl2=(N2//bl_len)-0
n_bl3=(N3//bl_len)-0
n_bl=n_bl1+n_bl2+n_bl3
tst_arr=[[20000] * bl_len for i in range(n_bl)]
tst_lab=[0 for i in range(n_bl)]

print(len(tst_lab))
for i in range(n_bl1):
    tst_lab[i]=0
    for j in range(bl_len):
        decod = struct.unpack("<hh", f1.readframes(1))
        tst_arr[i][j]=decod[1]
for i in range(n_bl2):
    ii=i+n_bl1
    tst_lab[ii]=1
    for j in range(bl_len):
        decod = struct.unpack("<hh", f2.readframes(1))
        tst_arr[ii][j]=decod[1]
for i in range(n_bl3):
    iii=i+n_bl1+n_bl2
    tst_lab[iii]=2
    for j in range(bl_len):
        decod = struct.unpack("<hh", f3.readframes(1))
        tst_arr[iii][j]=decod[1]
print(decod[1])   
tst_arr_numpy = array(tst_arr )
tst_lab_numpy=array(tst_lab)
print(tst_arr_numpy)
print(tst_arr_numpy.shape)
print(tst_lab_numpy)
print(tst_lab_numpy.shape)
f1.close()
f2.close()
f3.close()
#tst_arr_numpy = tst_arr_numpy.reshape((476, 1000, 1)) # CNN
tst_arr_numpy = tst_arr_numpy.reshape((n_bl, bl_len,1))
tst_arr_numpy = tst_arr_numpy.astype('float32') / 32000
############################################################################################

network = models.Sequential() 
#network.add(layers.Dense(1000, activation='relu', input_shape=(bl_len,))) 
#network.add(layers.Dense(256, activation='relu'))
#network.add(layers.Dense(128, activation='relu'))
#network.add(layers.Dropout(0.5))
#network.add(layers.Dense(64, activation='relu'))
#network.add(layers.Dropout(0.25))
#network.add(layers.Dense(3, activation='softmax'))


network.add(layers.Conv1D(512, (3), activation='relu', input_shape=(1000, 1)))
network.add(layers.MaxPooling1D(2))
network.add(layers.Conv1D(128, (3), activation='relu'))
network.add(layers.MaxPooling1D(2))
network.add(layers.Dropout(0.5))
network.add(layers.Conv1D(256, (3), activation='relu'))
network.add(layers.MaxPooling1D(2))
network.add(layers.Dropout(0.25))
network.add(layers.Flatten())
network.add(layers.Dense(256, activation='relu'))
network.add(layers.Dense(3, activation='softmax'))


network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.utils import to_categorical
ed_lab_numpy = to_categorical(ed_lab_numpy) 
tst_lab_numpy = to_categorical(tst_lab_numpy)

history = network.fit(ed_arr_numpy, ed_lab_numpy, validation_data=(ed_arr_numpy, ed_lab_numpy), epochs=10, batch_size=8) 

test_loss, test_acc = network.evaluate(tst_arr_numpy, tst_lab_numpy)
print('test_acc:', test_acc)
#########################################################################################
acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.title('Training accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()
