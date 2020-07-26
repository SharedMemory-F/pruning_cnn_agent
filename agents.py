import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D,Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
seed=1013
def tloss(y_true,y_pred):
  y_pred=tf.math.abs(y_pred)

  l1=tf.keras.losses.binary_crossentropy(y_true,y_pred)
  l2=tf.keras.losses.binary_crossentropy((tf.reduce_max(y_true)*tf.ones_like(y_true)-y_true),1-y_pred)
  return l1+l2

def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

class agent():
  def __init__(self,state_shape,action_cnt,lr=1e-3,factor=0.001):
    super(agent,self).__init__()
    self.state_shape=state_shape
    self.action_cnt=action_cnt
    self.lr=lr
    self.factor=factor
    self.states,self.actions,self.rewards=[],[],[]
    self.model=self._model()
    self.epoch=0
    self.lrschedular=LearningRateScheduler(scheduler)

  def _model(self):
    '''
    the model takes the filter matrix Wl=NlxMlxhxw as input
    and Wl will be arranged to 2D format, Wl=NlxMl
    if Ml is larger than 16, the alternating cnn with 7x7 kernel will be included
    other wise the pruning agent consist of two fc layer
    '''
    model=Sequential([
      Conv2D(32,(7,7),activation='relu',data_format="channels_first",padding="same",input_shape=self.state_shape),
      MaxPool2D(),
      Conv2D(64,(7,7),activation='relu',data_format="channels_first",padding="same"),
      MaxPool2D(),
      Conv2D(64,(7,7),activation='relu',data_format="channels_first",padding="same"),
      MaxPool2D(),
      Conv2D(64,(7,7),activation='relu',data_format="channels_first",padding="same"),
      MaxPool2D(),
      Flatten(),
      Dense(32,activation='relu'),
      Dense(32,activation='relu'),
      Dense(self.action_cnt,activation='sigmoid')
    ])
    model.compile(loss=tloss,optimizer=Adam(learning_rate=self.lr))
    return model
  def append_samples(self,reward,action,feature):
    self.states.append(feature)
    self.rewards.append(reward)
    self.actions.append(action)
  def get_action(self,feature):
    #binary the output to 0 and 1
    return self.model.predict(feature)[0]
  def discount_rewards(self,rewards):
    #the early rewards will have discounts
    discounted_rewards=np.zeros_like(rewards)
    c=0
    for t in range(len(rewards)):
      c=c*self.factor+rewards[-t]
      discounted_rewards[-t]=c
      print(c,end=" ")
    discounted_rewards-=np.mean(discounted_rewards)
    discounted_rewards/=np.std(discounted_rewards)
    return discounted_rewards
  def run(self):
    rewards=self.discount_rewards(self.rewards)
    episode_len=len(self.states)
    advantages=np.zeros((episode_len,self.action_cnt))
    
    h,w,c=self.state_shape
    inputs=np.zeros((episode_len,h,w,c))
    #episode is equal to action_cnt
    for i in range(episode_len):
      advantages[i][self.actions[i]]=rewards[i]
      inputs[i]=self.states[i]
    lr=self.lr*np.exp(-self.epoch//10)
    self.model.fit(inputs,advantages,epochs=self.epoch+3,initial_epoch=self.epoch,callbacks=[self.lrschedular])
    self.epoch+=3

    self.states,self.actions,self.rewards=[],[],[]
