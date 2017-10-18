#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D

from server import client_generator


def gen(hwm, host, port):
  print "----------------------start gen---------------------\n",hwm,"\n",host,"\n",port
  for tup in client_generator(hwm=hwm, host=host, port=port):
    X, Y, _ = tup
    Y = Y[:, -1]
    if X.shape[1] == 1:  # no temporal context
      X = X[:, -1]
    yield X, Y

'''
def get_model(time_len=1):
  ch, row, col = 3, 160, 320  # camera format

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model
'''

def get_model(time_len=1):
  ch, row, col = 3, 160, 320  # camera format

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
  model.add(Convolution2D(6, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Activation('relu'))
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Activation('relu'))
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Activation('relu'))
  model.add(Convolution2D(48, 3, 3, subsample=(1, 1), border_mode="same"))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.5))
  model.add(Activation('relu'))
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(Activation('relu'))
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  model = get_model()
  traindata = gen(20, args.host, port=args.port)
  validationdata = gen(20, args.host, port=args.val_port)
  step_per_epoch_train = 50
  steps_batch_valid = 10
    
  '''
  model.fit_generator(
      traindata,
      steps_per_epoch=step_per_epoch_train,
      epochs=args.epoch
     )
  '''
  model.fit_generator(
    traindata,
    samples_per_epoch=10000,
    nb_epoch=args.epoch,verbose = 1, show_accuracy = True,
    validation_data=validationdata,
    nb_val_samples=1000
  )

  
    
  #model.predict_generator(gen(20, args.host, port=args.val_port),500)
  

  #model.evaluate_generator(gen(20, args.host, port=args.val_port),500)
  print("Saving model weights and configuration file.")

  if not os.path.exists("/data/self_driving/outputs/steering_model"):
      os.makedirs("/data/self_driving/outputs/steering_model")

  model.save_weights("/data/self_driving/outputs/steering_model/steering_angle.keras", True)
  with open('/data/self_driving/outputs/steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
  print "\ntest!!    ",dir(traindata)
  #print "\ntest!!    ",gen(20, args.host, port=args.port).sizeof()
  '''
  j=0
  for i in gen(20, args.host, port=args.val_port):
    print("ii--!--ii",j)
    print(i)
    j = j+1
   '''
  print "\ntest!!    ",traindata.__sizeof__()