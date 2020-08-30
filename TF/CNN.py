import tensorflow as tf
import Model_CNN as model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# double化，因为double范围为0~1
x_train, x_test = x_train / 255.0, x_test / 255.0
# 数据维度预处理
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 不是很稳定,即有时候不能收敛
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.0,nesterov=False)
# 稳定、收敛快且精度高于SGD
optimizer = tf.keras.optimizers.Adam(lr=0.01)
loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)



# 设置回调功能
filepath = 'model/CNN_model' # 保存模型地址
# saved_model = tf.keras.callbacks.ModelCheckpoint(filepath, verbose = 1) # 回调保存模型功能
# tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'model/log') # 回调可视化数据功能

model = model.Model_CNN()
model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

model.fit(train_ds, batch_size=32, epochs=5) # callbacks = [saved_model, tensorboard]

