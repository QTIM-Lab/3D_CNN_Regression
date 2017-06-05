import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, Dropout
from keras.optimizers import Adam
from keras.models import load_model
from keras import regularizers

from dummy_data import dummy_data_generator

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

def regression_model_3d(input_shape, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, convolutions=4, dropout=.25, filter_shape=(3,3,3), deconvolution=True):

    inputs = Input(input_shape)
    # inputs = Input((1,32,32,32,4))
    # conv_mid = Conv3D(int(32/downsize_filters_factor), filter_shape, activation='relu', padding='same', data_format='channels_first')(inputs)

    # conv_mid = Dropout(0.25)(conv_mid)

    # for conv_num in xrange(convolutions-2):

    #     conv_mid = Conv3D(int(32/downsize_filters_factor), filter_shape, activation='relu', padding='same', data_format='channels_first')(conv_mid)
    #     conv_mid = Dropout(0.25)(conv_mid)

    # conv_out = Conv3D(int(1), filter_shape, activation='tanh', padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(0.01))(conv_mid)

    # model = Model(inputs=inputs, outputs=conv_out)

    # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq])

    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3), activation='relu', data_format='channels_first',
                   padding='same')(inputs)
    conv1 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu', data_format='channels_first',
                   padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size, data_format='channels_first',)(conv1)

    conv2 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(pool1)
    conv2 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size, data_format='channels_first')(conv2)

    conv3 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(pool2)
    conv3 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size, data_format='channels_first')(conv3)

    conv4 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(pool3)
    conv4 = Conv3D(int(512/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(conv4)

    up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2,
                     nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(conv4)
    up5 = concatenate([up5, conv3], axis=1)
    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu', data_format='channels_first',padding='same')(up5)
    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(conv5)

    up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                     nb_filters=int(256/downsize_filters_factor),image_shape=input_shape[-3:])(conv5)
    up6 = concatenate([up6, conv2], axis=1)
    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first', padding='same')(up6)
    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same')(conv6)

    up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                     nb_filters=int(128/downsize_filters_factor), image_shape=input_shape[-3:])(conv6)
    up7 = concatenate([up7, conv1], axis=1)
    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first', padding='same')(up7)
    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                   padding='same', kernel_regularizer=regularizers.l2(0.01))(conv7)

    conv8 = Conv3D(int(1), (1, 1, 1), data_format='channels_first',)(conv7)
    model = Model(inputs=inputs, outputs=conv8)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq])

    return model

def msq(y_true, y_pred):
    return K.sum(K.pow(y_true - y_pred, 2), axis=None)

def msq_loss(y_true, y_pred):
    return msq(y_true, y_pred)

def load_old_model(model_file):
    print("Loading pre-trained model")

    custom_objects = {'msq': msq, 'msq_loss': msq_loss}

    try:
        from keras_contrib.layers import Deconvolution3D
        custom_objects["Deconvolution3D"] = Deconvolution3D
    except ImportError:
        print("Could not import Deconvolution3D. To use Deconvolution3D install keras-contrib.")

    return load_model(model_file, custom_objects=custom_objects)

def compute_level_output_shape(filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    if depth != 0:
        output_image_shape = np.divide(image_shape, np.multiply(pool_size, depth)).tolist()
    else:
        output_image_shape = image_shape
    return tuple([None, filters] + [int(x) for x in output_image_shape] )

def get_upconv(depth, nb_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2),
               deconvolution=False):

    print compute_level_output_shape(filters=nb_filters, depth=depth+1, pool_size=pool_size, image_shape=image_shape)

    if deconvolution and False:
        try:
            from keras_contrib.layers import Deconvolution3D
        except ImportError:
            raise ImportError("Install keras_contrib in order to use deconvolution. Otherwise set deconvolution=False.")

        return Deconvolution3D(filters=nb_filters, kernel_size=kernel_size,
                               output_shape=compute_level_output_shape(filters=nb_filters, depth=depth,
                                                                       pool_size=pool_size, image_shape=image_shape),
                               strides=strides, input_shape=compute_level_output_shape(filters=nb_filters,
                                                                                       depth=depth+1,
                                                                                       pool_size=pool_size,
                                                                                       image_shape=image_shape))
    else:
        return UpSampling3D(size=pool_size, data_format='channels_first')

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