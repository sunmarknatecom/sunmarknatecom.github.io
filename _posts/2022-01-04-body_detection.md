```python
from PIL import Image
import numpy as np
import glob
import tensorflow as tf
import cv2
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
from IPython.display import clear_output

# file_list = sorted(glob.glob("*.png"))

# temp_list = []

# for fn in file_list:
#     if fn[:3] == "new":
#         temp_list.append(fn)
#     else:
#         pass

# for fn in temp_list:
#     file_obj = Image.open(fn)
#     arr_obj = np.array(file_obj)
#     ht, wd = np.shape(arr_obj)
#     for h in range(ht):
#         for w in range(wd):
#             if arr_obj[h][w] == 255:
#                 arr_obj[h][w] = 2
#             else:
#                 arr_obj[h][w] = 1
#     temp_img = Image.fromarray(arr_obj)
#     temp_img.save(fn[4:9]+"_mask.png","PNG")


# newImg1.save("img1.png","PNG")

TRAIN_DS = sorted(glob.glob("*.jpg"))
MASK__DS = sorted(glob.glob("*.png"))

TRAIN_DATASET = []
MASK__DATASET = []

for fn in TRAIN_DS:
    temp_obj = Image.open(fn)
    temp2_obj = np.array(temp_obj)
    arr_obj = cv2.resize(temp2_obj, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    TRAIN_DATASET.append(arr_obj)

for fn in MASK__DS:
    temp_obj = Image.open(fn)
    temp2_obj= np.array(temp_obj)
    arr_obj = cv2.resize(temp2_obj, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    MASK__DATASET.append(arr_obj)

TRAIN_TEMP = []
for elem in TRAIN_DATASET:
    input_image = tf.cast(elem, tf.float32) / 255.0
    TRAIN_TEMP.append(input_image)

MASK_TEMP = []
for elem in MASK__DATASET:
    temp = elem -1
    MASK_TEMP.append(temp)

# IMG_DS = tf.constant(TRAIN_TEMP)
# MSK_DS = tf.constant(MASK_TEMP)

img_dataset = tf.data.Dataset.from_tensor_slices((TRAIN_TEMP, MASK_TEMP))

TRAIN_LENGTH = 307
BATCH_SIZE = 32
BUFFER_SIZE = 100
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

img_dataset = img_dataset.batch(BATCH_SIZE)
img_dataset = img_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()

for image, mask in img_dataset.take(1):
    sample_image, sample_mask = image, mask

display([sample_image[0], sample_mask[0]])

OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

layer_names = ['block_1_expand_relu', 'block_3_expand_relu','block_6_expand_relu','block_13_expand_relu', 'block_16_project',]

layers = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [pix2pix.upsample(512, 3), pix2pix.upsample(256,3), pix2pix.upsample(128, 3), pix2pix.upsample(64, 3)]

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2,padding='same')  #64x64 -> 128x128
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\n에포크 이후 예측 예시 {}\n'.formate(epoch+1))


EPOCHS = 20
VAL_SUBSPLITS = 5

model_history = model.fit(img_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[DisplayCallback()])
```
