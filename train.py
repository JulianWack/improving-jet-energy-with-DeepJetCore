from DeepJetCore.training.training_base import training_base
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, BatchNormalization, Concatenate #etc

from Losses import my_loss
from Layers import GravNet_simple


def my_model(Inputs,otheroption):

    x = Inputs[0] # feature array

    GravNet_layer1 = GravNet_simple(n_propagate = 64, n_dimensions = 3, n_neighbours = 15, n_filters = 64)
    GravNet_layer1.build(x.shape)
    x = GravNet_layer1.call(x)

    GravNet_layer2 = GravNet_simple(n_propagate = 32, n_dimensions = 3, n_neighbours = 15, n_filters = 32)
    GravNet_layer2.build(x.shape)
    x = GravNet_layer2.call(x)

    #x = Dense(8, use_bias=False)(x)

    #GravNet_layer2 = GravNet_simple(n_propagate = 8, n_dimensions = 3, n_neighbours = 10, n_filters = 16)
    #GravNet_layer2.build(x.shape)
    #x = GravNet_layer2.call(x)

    x = Dense(8, use_bias=False)(x)

    predictions = Concatenate()([x, Inputs[1]]) # Inputs[1] is PF pt

    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=False,renewtokens=False)

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    train.setModel(my_model,otheroption=1)

    train.compileModel(learningrate=1e-5,
                   loss=my_loss)

print(train.keras_model.summary())


model,history = train.trainModel(nepochs=30,
                                 batchsize=5,
                                 checkperiod=5, # saves a checkpoint model every N epochs
                                 verbose=1)

print('Since the training is done, use the predict.py script to predict the model output on your test sample,'\
      'e.g.: predict.py <training output>/KERAS_model.h5 <training output>/trainsamples.djcdc <your subpackage>/example_data/test_data.txt <output dir>')
