import tensorflow as tf
from PIL import Image
import os
import numpy as np
#
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # 对输入特征进行归一化，将以前0-255之间的灰度值，变为0-1之间的数值，
                                                    # 将输入的数值变小，更适合神经网络的吸收

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()


# '''
#     input: 图片路径， 标签文件
#     return: 输入特征和标签
# '''
# def generateds(path, txt):
#     f = open(txt, 'r')  # 以只读形式打开txt文件
#     contents = f.readlines()  # 读取文件中所有行
#     f.close()
#
#     x, y_ = [], []
#     for content in contents:  # 逐行读出，以空格分开
#         value = content.split()
#         img_path = path + value[0]  # 图片名为value[0]
#         img = Image.open(img_path)
#         img = np.array(img.convert('L'))   # 图片变为8位跨宽度的灰度值
#         img = img / 255.0
#         x.append(img)
#         y_.append(value[1])
#         print("loading : " + content)
#
#     x = np.array(x)
#     y_ = np.array(y_)
#     y_ = y_.astype(np.int64)
#     return x, y_
