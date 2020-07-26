import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam,SGD,schedules
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.callbacks import LearningRateScheduler,TensorBoard,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input,GlobalAveragePooling2D
from tensorflow.keras import Model
import numpy as np
import shutil
import os

from environments import environment
from agents import agent
class Average():
    def __init__(self):
        super().__init__()
        self.value=None
        self.n=0
        self.sum=None
        self.average=None
    def update(self,v,cnt=1):
        self.value=v
        self.n+=cnt
        if self.sum is not None:
            self.sum+=v
        else:
            self.sum=v
        self.average=self.sum/self.n
if __name__ == "__main__":
    seed=1013
    '''
    use cifar with 10 classes,normalize to 0-1
    '''
    (x_train,y_train),(x_test,y_test)=keras.datasets.cifar10.load_data()
    x_train=x_train.astype('float32')/256
    x_test=x_test.astype('float32')/256
    y_train=keras.utils.to_categorical(y_train,num_classes=10)
    y_test=keras.utils.to_categorical(y_test,num_classes=10)
    '''
    train a vgg16 to cifar10
    '''
    batch_size=256
    steps_per_epoch=len(x_train)//batch_size
    seed=1013
    valid_steps=len(x_test)//batch_size
    train_datagen=ImageDataGenerator(
        rotation_range=10,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.2
    )
    train_gen=train_datagen.flow(x_train,y_train,batch_size=batch_size,seed=seed)
    valid_datagen=ImageDataGenerator()
    valid_gen=valid_datagen.flow(x_test,y_test,batch_size=batch_size,seed=seed)
    vgg_model = VGG16(input_shape=x_train[0].shape,classes=10,include_top=False)
    vgg_model.run_eagerly=False
    flatten=GlobalAveragePooling2D()(vgg_model.output)
    predict=Dense(10,activation="softmax")(flatten)
    vgg16=Model(inputs=vgg_model.input,outputs=predict)
    check_point_path="./vgg_check_point.hdf5"
    model_path="vgg_model.hdf5"
    check_point=ModelCheckpoint(check_point_path,save_best_only=True,monitor='val_accuracy')
    tensorboard=TensorBoard(log_dir="logs/vgg")
    epochs=10
    if os.path.exists(check_point_path):
        vgg16.load_weights(check_point_path)
    vgg16.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),metrics=["accuracy"])
    vgg16_acc=vgg16.evaluate(valid_gen,steps=valid_steps)[1]
    if vgg16_acc<0.8:
        vgg16.fit(train_gen,steps_per_epoch=steps_per_epoch,epochs=epochs,validation_data=valid_gen,validation_steps=valid_steps,callbacks=[check_point,tensorboard])
    '''
    prune on vgg16, the layer index can be found by model.summary()
    '''
    state=0
    base_model=vgg16
    base_model.evaluate(valid_gen,steps=valid_steps)
    new_model=Model(inputs=vgg_model.input,outputs=predict)
    layer_index=17
    action_cnt=base_model.layers[layer_index].get_weights()[0].shape[-1]
    h=base_model.layers[layer_index].get_weights()[0].shape[:3]
    print("layer {} have {} filters".format(layer_index,action_cnt))

    ag=agent((action_cnt,h[0]*h[1]*h[2],1),action_cnt,lr=1e-3,factor=0.01)
    env=environment(base_model,new_model,x_train,y_train,x_test,y_test,steps_per_epoch,valid_steps,batch_size=batch_size,b=0.05,layer_index=layer_index,epochs=1)
    episode_len=5
    writer = tf.summary.create_file_writer("./logs/mylogs_7")
    Avg_prob=Average()
    action_choice=None
    with writer.as_default():
        for i in range(100):
            feature=env.layer_weights
            feature=feature.reshape(1,action_cnt,-1,1)
            action_prob=ag.get_action(feature)


            Avg_prob.update(action_prob)
            Avg_reward=Average()

            for ep in range(episode_len):
                redundant_channels,actions,reward,acc=env.run(action_prob)
                ag.append_samples(reward,redundant_channels,feature)
                print("state {}, prune {} filer from {},accuracy {}/{},reward {}".format(state,len(redundant_channels),action_cnt,acc,env.base_model_acc,reward))
                Avg_reward.update(reward)
                state+=1
            tf.summary.scalar("avg_reward", Avg_reward.average,step=i)
            tf.summary.scalar("prob_distance", np.sum(np.abs(action_prob-Avg_prob.average)),step=i)
            if action_choice is None:
                action_choice=action_prob>0.5
            else:
                if (action_choice==(action_prob>0.5)).all():
                    break
                else:
                    action_choice=action_prob
            ag.run()
            writer.flush()
