from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten,Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np
import copy
seed=1013
class environment:
  def __init__(self,base_model,new_model,x_train,y_train,x_test,y_test,train_steps,valid_steps,batch_size=128,b=0.05,layer_index=0,epochs=2):
    self.base_model=base_model
    self.new_model=new_model
    train_datagen=ImageDataGenerator(
        rotation_range=10,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.2
    )
    train_gen=train_datagen.flow(x_train,y_train,batch_size=batch_size,seed=seed)
    valid_datagen=ImageDataGenerator()
    valid_gen=valid_datagen.flow(x_test,y_test,batch_size=batch_size,seed=seed)
    self.train_gen=train_gen
    self.valid_gen=valid_gen
    self.valid_steps=valid_steps
    self.train_steps=train_steps
    self.b=b
    self.epochs=epochs
    self.layer_index=layer_index
    self.base_model_acc=self.accuracy(self.base_model)
    self.layer=self.get_layer()
    self.layer_weights=self.layer.get_weights()[0]
    self.layer_bias=self.layer.get_weights()[1]
    self.best_model_acc=self.base_model_acc

  def accuracy(self,model):
    return model.evaluate(self.valid_gen,steps=self.valid_steps)[1]
  def get_layer(self):
    try:
      #in some version of tf,Module can't use get_layer
      return self.base_model.get_layer(index=self.layer_index)
    except:
      return self.base_model.layers[self.layer_index]

  def reward(self,new_model,actions):
    new_model.fit(self.train_gen,steps_per_epoch=self.train_steps,epochs=self.epochs,validation_data=self.valid_gen,validation_steps=self.valid_steps)
    new_model_acc=self.accuracy(new_model)
    return (1-(self.base_model_acc-new_model_acc)/self.b)*np.log(actions.shape[0]/(0.0001+np.sum(actions))),new_model_acc
  def prune_layer(self,channels):
    new_weights=self.layer_weights.copy()
    for channel in channels:
      new_weights[:,:,:,channel]=0
    return new_weights
  def get_actions(self,action_prob):
    epison=0.01
    choice=np.zeros_like(action_prob)
    rand=np.random.random_sample(size=choice.shape[0])
    action_prob_copy=action_prob.copy()
    action_prob_copy-=np.min(action_prob)
    action_prob_copy+=epison
    action_prob_copy/=(np.max(action_prob))
    choice=action_prob_copy>rand
    return choice
  def run(self,action_prob):
    actions=self.get_actions(action_prob)
    redundant_channels=np.where(actions==0)[0]#where actions=0 will be pruned
    #surgery f according to A_i
    new_weights=self.prune_layer(redundant_channels)
    new_model=self.new_model
    new_model.set_weights(self.base_model.get_weights().copy())
    new_model.layers[self.layer_index].set_weights([new_weights,self.layer_bias])
    new_model.layers[self.layer_index].trainable=False
    print(new_model.layers[self.layer_index])
    #fintune using X_train
    new_model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),metrics=["accuracy"])
    reward,acc=self.reward(new_model,actions)
    if acc>self.best_model_acc:
      self.best_model_acc=acc
      print(new_model.layers[self.layer_index])
      new_model.save("vgg_check_point_pruned.hdf5")
    return redundant_channels,actions,reward,acc