from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization

# 生成数据集
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32 # batch size
TS=(24,24) # target size
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
# 每个epoch的步数；train_batch.classes获取labels
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)

# 构建模型
model = Sequential([
    # 第一层 32个卷积滤波器 每个大小3×3
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),   # 第一层卷积层
    MaxPooling2D(pool_size=(1,1)),  # 第一层池化层

    # 第二层 32个卷积滤波器 每个大小3×3
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

    # 第三层 64个卷积滤波器 每个大小 3x3
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

    # 全连接层
    Dropout(0.25), # 随机失活
    Flatten(), # 将多维输入一维化
    Dense(128, activation='relu'), # 全连接层
    Dropout(0.5), # 随机失活使模型更好地收敛

    # 输出层Fully Connected Layer
    Dense(2, activation='softmax')
])
# 编译模型 Adam优化器、交叉熵损失函数和准确率作为评估指标
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# 使用生成器进行模型训练
model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)
# 保存模型
model.save('models/cnnCat2.h5', overwrite=True)