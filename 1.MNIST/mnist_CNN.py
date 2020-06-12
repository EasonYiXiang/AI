# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:15:07 2020

@author: Eason
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



# Add a channels dimension
# 多增加第4個 dimension:"channel"
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(x_train.shape)


#== tf.data API 來建立 input pipeline ==#
#  * tf.data

#1."from_tensor_slices" 建立一個 Dataset Object
#  在使用"from_tensor_slices"參數（x_train, y_train）是
#  包含很多筆類型一樣的data,像這個例子有 60000筆data。
#  不要跟"from_tensor"搞混了!

#2.將training data做shuffle打亂 和 將batch size設定為 32
# tf.data.Dataset.from_tensor_slices : Creates a Dataset whose elements are slices of the given tensors.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(32)


#== 使用 Keras 自訂一個 Model ==#
#  * tf.Keras.Model
#  * tf.Keras.layers

#1.建立一個 Class並繼承 tf.Keras.Model
#2.__init__裡面我們可以自己設計每一層 Layer長什麼樣子，
#  是哪種 Layer(Convolution, dense...)

#3.call定義了這個 Model的 forward pass
#
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()     # call __init__() from tf.Keras.Model
    # Define your layers here.
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    # Define your forward pass here,
    # using layers you previously defined (in __init__).
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

model = MyModel()


#== 選擇 loss, optimizer和 metrics ==#
#  * tf.keras.losses
#  * tf.keras.optimizers
#  * tf.keras.metrics


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#== train the model (定義)==#
#  * tf.function
#  * tf.GradientTape

# @ : Decorator 
@tf.function
def train_step(images, labels):
    
    #
    with tf.GradientTape() as tape:
        # forward pass to get predictions
        predictions = model(images)
        # compute loss with the ground and our predictions
        loss = loss_object(labels, predictions)
    # compute gradient 
    gradients = tape.gradient(loss, model.trainable_variables)
    # back propagation
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    
   
    
#== test the model (定義)==#
#  * tf.function

@tf.function
def test_step(images, labels):
    
    # forward pass to get predictions
    predictions = model(images)
    # compute loss with the ground and our predictions
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
    
    
#== 真正開始 train model ==#

# 定義EPOCH數(整個dataset 拿來train過一次稱一個epoch)
EPOCH = 5
for epoch in range(EPOCH):
    for images, labels in train_dataset:
        train_step(images, labels)

    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)    
    
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}\n'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()