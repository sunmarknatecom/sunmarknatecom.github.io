```python
!pip list
```


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Input
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
```


```python
base_path = '/kaggle/input/ranzcr-clip-catheter-line-classification/'

for path in ['train','test']:
    print('{} data size:{}'.format(path,len(os.listdir(os.path.join(base_path,path)))))

df_train = pd.read_csv(os.path.join(base_path,'train.csv'))
df_test = pd.read_csv(os.path.join(base_path,'sample_submission.csv'))

Labels = np.array(df_train.drop(['StudyInstanceUID','PatientID'],axis=1).columns)

print('train_csv shapes',df_train.shape)
print()
print('Labels:',Labels)
```

    train data size:30083
    test data size:3582
    train_csv shapes (30083, 13)
    
    Labels: ['ETT - Abnormal' 'ETT - Borderline' 'ETT - Normal' 'NGT - Abnormal'
     'NGT - Borderline' 'NGT - Incompletely Imaged' 'NGT - Normal'
     'CVC - Abnormal' 'CVC - Borderline' 'CVC - Normal'
     'Swan Ganz Catheter Present']
    


```python
display(df_train.tail(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StudyInstanceUID</th>
      <th>ETT - Abnormal</th>
      <th>ETT - Borderline</th>
      <th>ETT - Normal</th>
      <th>NGT - Abnormal</th>
      <th>NGT - Borderline</th>
      <th>NGT - Incompletely Imaged</th>
      <th>NGT - Normal</th>
      <th>CVC - Abnormal</th>
      <th>CVC - Borderline</th>
      <th>CVC - Normal</th>
      <th>Swan Ganz Catheter Present</th>
      <th>PatientID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30073</th>
      <td>1.2.826.0.1.3680043.8.498.44675490137018694724...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>b32884471</td>
    </tr>
    <tr>
      <th>30074</th>
      <td>1.2.826.0.1.3680043.8.498.60620865844062547094...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>45d82b916</td>
    </tr>
    <tr>
      <th>30075</th>
      <td>1.2.826.0.1.3680043.8.498.12112840402677606176...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>cccbc15ba</td>
    </tr>
    <tr>
      <th>30076</th>
      <td>1.2.826.0.1.3680043.8.498.59704742952729813362...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>172c3c7ed</td>
    </tr>
    <tr>
      <th>30077</th>
      <td>1.2.826.0.1.3680043.8.498.97304417279653947772...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>b304abf90</td>
    </tr>
    <tr>
      <th>30078</th>
      <td>1.2.826.0.1.3680043.8.498.74257566841157531124...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5b5b9ac30</td>
    </tr>
    <tr>
      <th>30079</th>
      <td>1.2.826.0.1.3680043.8.498.46510939987173529969...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7192404d8</td>
    </tr>
    <tr>
      <th>30080</th>
      <td>1.2.826.0.1.3680043.8.498.43173270582850645437...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>d4d1b066d</td>
    </tr>
    <tr>
      <th>30081</th>
      <td>1.2.826.0.1.3680043.8.498.95092491950130838685...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>01a6602b8</td>
    </tr>
    <tr>
      <th>30082</th>
      <td>1.2.826.0.1.3680043.8.498.99518162226171269731...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>e692d316c</td>
    </tr>
  </tbody>
</table>
</div>



```python
train_path =  str(base_path + '/train')

rows = 3
cols = 3

display_size = rows*cols
files = zip(df_train['StudyInstanceUID'][:display_size]+'.jpg',df_train.drop(['StudyInstanceUID','PatientID'],axis=1).to_numpy()[:display_size])

plt.figure(figsize=(15,10))

for i, (file_name,label) in enumerate(files):
    plt.subplot(rows,cols,i+1)
    img = cv2.imread(os.path.join(train_path,file_name),0)
    plt.imshow(img, cmap='gray')
    plt.title(Labels[[lbl[0] for lbl in np.argwhere(label==1)]].tolist())
    plt.ylabel('img_size='+str(img.shape))
plt.show()
```


    
![png](output_4_0.png)
    



```python
def path2img(path, label = None, resize = (224,224), mode_rgb = 0):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels = 3)
    if mode_rgb == 0:
        img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img,resize)
    img = tf.cast(img,tf.float32)/255.0
    return img, label if label!=None else img


def process_dataset(files, training_data):
    A = tf.data.experimental.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    files_ds = files_ds.map(path2img, num_parallel_calls=A)
    if training_data:
        files_ds = files_ds.cache()
    return files_ds
```


```python
paths = base_path + 'train/' + df_train['StudyInstanceUID']+'.jpg'
labels = df_train.drop(['StudyInstanceUID','PatientID'],axis=1).to_numpy()

x_train_path, x_val_path, y_train, y_val = train_test_split(paths, labels, test_size = .2, random_state=42, shuffle = True)

print('train_size:',len(y_train))
print('val_size:',len(y_val))

train = process_dataset((x_train_path,y_train), training_data = True)
val = process_dataset((x_val_path,y_val), training_data = True)

plt.figure(figsize=(12,6))
print('train samples:')
for i, (img,label) in enumerate(train.take(2)):
    plt.subplot(1,2,i+1)
    plt.imshow(img,cmap='gray')
    plt.title(Labels[[lbl[0] for lbl in np.argwhere(label==1)]].tolist())
    plt.xlabel('image_size:'+str(img.shape))
plt.show()

plt.figure(figsize=(12,6))
print('validation samples:')
for i, (img,label) in enumerate(val.take(2)):
    plt.subplot(1,2,i+1)
    plt.imshow(img,cmap='gray')
    plt.title(Labels[[lbl[0] for lbl in np.argwhere(label==1)]].tolist())
    plt.xlabel('image_size:'+str(img.shape))
plt.show()
```

    train_size: 24066
    val_size: 6017
    train samples:
    


    
![png](output_6_1.png)
    


    validation samples:
    


    
![png](output_6_3.png)
    



```python
input_shape = img.shape
print('input_shape:',input_shape)
num_classes = len(Labels)

model = Sequential()
model.add(Conv2D(64,kernel_size=(3,3), padding = 'same', input_shape = input_shape))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(128,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(256,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))         
model.add(MaxPooling2D(pool_size=(2,2)))   

model.add(Conv2D(512,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))         
model.add(Conv2D(512,kernel_size=(3,3), padding = 'same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.2))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(.2))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))
model.summary()

```

    input_shape: (224, 224, 1)
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 224, 224, 64)      640       
    _________________________________________________________________
    batch_normalization (BatchNo (None, 224, 224, 64)      256       
    _________________________________________________________________
    activation (Activation)      (None, 224, 224, 64)      0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 224, 224, 64)      36928     
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 224, 224, 64)      256       
    _________________________________________________________________
    activation_1 (Activation)    (None, 224, 224, 64)      0         
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 112, 112, 64)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 112, 112, 128)     73856     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 112, 112, 128)     512       
    _________________________________________________________________
    activation_2 (Activation)    (None, 112, 112, 128)     0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 112, 112, 128)     147584    
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 112, 112, 128)     448       
    _________________________________________________________________
    activation_3 (Activation)    (None, 112, 112, 128)     0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 56, 56, 128)       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 56, 56, 256)       295168    
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 56, 56, 256)       1024      
    _________________________________________________________________
    activation_4 (Activation)    (None, 56, 56, 256)       0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 56, 56, 256)       590080    
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 56, 56, 256)       224       
    _________________________________________________________________
    activation_5 (Activation)    (None, 56, 56, 256)       0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 28, 28, 256)       0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 28, 28, 512)       2048      
    _________________________________________________________________
    activation_6 (Activation)    (None, 28, 28, 512)       0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 28, 28, 512)       2048      
    _________________________________________________________________
    activation_7 (Activation)    (None, 28, 28, 512)       0         
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               6422784   
    _________________________________________________________________
    activation_8 (Activation)    (None, 256)               0         
    _________________________________________________________________
    dropout (Dropout)            (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                16448     
    _________________________________________________________________
    activation_9 (Activation)    (None, 64)                0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    activation_10 (Activation)   (None, 32)                0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 11)                363       
    _________________________________________________________________
    activation_11 (Activation)   (None, 11)                0         
    =================================================================
    Total params: 11,132,715
    Trainable params: 11,129,307
    Non-trainable params: 3,408
    _________________________________________________________________
    


```python
lr = 7e-4
epochs = 20
batch_size = 86

train_b = train.batch(batch_size)
val_b = val.batch(batch_size)
 
CP = ModelCheckpoint('/kaggle/working/model.hdf5', save_best_only=True, verbose = 1)

model.compile(optimizer = Adam(lr=lr), loss='binary_crossentropy', metrics=['AUC'])
H = model.fit(train_b, epochs = epochs, validation_data = val_b, callbacks=[CP], shuffle=True)
```

    Epoch 1/20
     49/280 [====>.........................] - ETA: 29:47 - loss: 1.2699 - auc: 0.6233


```python
plt.figure(figsize=(8,4))
plt.plot(H.history['auc'])
plt.plot(H.history['val_auc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()

plt.figure(figsize=(8,4))
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()
```


```python
del train_b
del val_b

test_path = base_path + 'test/' + df_test['StudyInstanceUID']+'.jpg'

test = process_dataset(test_path, training_data = False)

#load the best model saved
model = load_model('/kaggle/working/model.hdf5')

predictions = model.predict(test.batch(1), verbose = 1)

df_test.iloc[:,1:] = predictions

display(df_test.head())

df_test.to_csv('submission.csv', index = False)
```
