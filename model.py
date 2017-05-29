import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation
from keras.optimizers import Adam
from keras.models import load_model

from dummy_data import dummy_data_generator

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

def regression_model_3d(input_shape, downsize_filters_factor=1, initial_learning_rate=0.00001):

    # 144x144x144
    inputs = Input(input_shape)

    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(inputs)

    conv1 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv1)

    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv1)

    conv1 = Conv3D(int(1/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv1)

    model = Model(inputs=inputs, outputs=conv1)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq])

    return model

def msq(y_true, y_pred):
    return K.sum(K.pow(y_true - y_pred, 2), axis=None)

def msq_loss(y_true, y_pred):
    return -msq(y_true, y_pred)

def load_old_model(model_file):
    print("Loading pre-trained model")

    custom_objects = {'msq': msq, 'msq_loss': msq_loss}

    try:
        from keras_contrib.layers import Deconvolution3D
        custom_objects["Deconvolution3D"] = Deconvolution3D
    except ImportError:
        print("Could not import Deconvolution3D. To use Deconvolution3D install keras-contrib.")

    return load_model(model_file, custom_objects=custom_objects)

if __name__ == '__main__':

    pass

    # config["pool_size"] = (2, 2, 2)
    # config["image_shape"] = (144, 144, 144)  # This determines what shape the images will be cropped/resampled to.
    # config["n_labels"] = 1  # not including background

    # model = regression_model_3d((4, 5, 5, 5), downsize_filters_factor=1, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False)

    # case_num = 60
    # training_data = dummy_data_generator(cases=case_num, input_modalities=4, modality_dims=(5,5,5))




    # testing_data = dummy_data_generator(cases=test_num, input_modalities=4, modality_dims=(5,5,5))

    # output_model_file = 'test.h5'

    # train_model(model=model, model_file=output_model_file, training_generator=train_generator,validation_generator=validation_generator, steps_per_epoch=nb_train_samples, validation_steps=nb_test_samples, initial_learning_rate=config["initial_learning_rate"], learning_rate_drop=config["learning_rate_drop"], learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"])  