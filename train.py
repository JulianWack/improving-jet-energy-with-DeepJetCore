from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization #etc

from loss_function import my_loss

def my_model(Inputs,otheroption):

    x = Inputs[0] #this is the self.x list from the TrainData data structure
    x = Dense(1, activation='relu', use_bias=False)(x)
    # x = BatchNormalization(momentum=0.9)(x)
    # x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    # x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    # x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    # x = BatchNormalization(momentum=0.9)(x)
    # x = Conv2D(8,(4,4),strides=(2,2),activation='relu', padding='valid')(x)
    # x = Conv2D(4,(4,4),strides=(2,2),activation='relu', padding='valid')(x)
    # x = Flatten()(x)
    # x = Dense(32, activation='relu')(x)

    # 3 prediction classes
    # x = Dense(3, activation='softmax')(x)

    predictions = [x, Inputs[1]]
    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=False,renewtokens=False)

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    train.setModel(my_model,otheroption=1)

    train.compileModel(learningrate=0.03,
                   loss=my_loss)

print(train.keras_model.summary())


model,history = train.trainModel(nepochs=5,
                                 batchsize=2,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1)

print('Since the training is done, use the predict.py script to predict the model output on your test sample,'\
      'e.g.: predict.py <training output>/KERAS_model.h5 <training output>/trainsamples.djcdc <your subpackage>/example_data/test_data.txt <output dir>')
