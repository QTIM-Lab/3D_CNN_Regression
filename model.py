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

def n_net_3d(input_shape, output_shape, initial_convolutions_num=3, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, dropout=.25, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True):

    # Convenience variables.
    # For now, we assume that the modalties are ordered by nesting priority.
    output_modalities = output_shape[0]

    # Original input
    inputs = Input(input_shape)

    # Change the space of the input data into something a bit more generalized using consecutive convolutions.
    initial_conv = Conv3D(int(8/downsize_filters_factor), filter_shape, activation='relu', padding='same', data_format='channels_first')(inputs)
    initial_conv = Dropout(dropout)(initial_conv)
    if initial_convolutions_num > 1:
        for conv_num in xrange(initial_convolutions_num-1):

            initial_conv = Conv3D(int(8/downsize_filters_factor), filter_shape, activation='relu', padding='same', data_format='channels_first')(initial_conv)
            initial_conv = Dropout(dropout)(initial_conv)

    # Cascading U-Nets
    input_list = [initial_conv] * output_modalities
    output_list = [None] * output_modalities
    for modality in xrange(output_modalities):

        for output in output_list:
            if output is not None:
                input_list[modality] = concatenate([input_list[modality], output], axis=1)

        print '\n'
        print 'MODALITY', modality, 'INPUT LIST', input_list[modality]
        print '\n'

        output_list[modality] = u_net_3d(input_shape=input_shape, input_tensor=input_list[modality], downsize_filters_factor=downsize_filters_factor*4, pool_size=(2, 2, 2), initial_learning_rate=initial_learning_rate, dropout=dropout, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True)

    # Concatenate results
    print output_list
    final_output = output_list[0]
    if len(output_list) > 1:
        for output in output_list[1:]:
            final_output = concatenate([final_output, output], axis=1)

    # Get cost
    if regression:
        act = Activation('relu')(final_output)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq])
    else:
        act = Activation('sigmoid')(final_output)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def w_net_3d(input_shape, output_shape, initial_convolutions_num=3, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, dropout=.25, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True):

    # Convenience variables.
    # For now, we assume that the modalties are ordered by nesting priority.
    output_modalities = output_shape[0]

    inputs = Input(input_shape)

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

    input_list = [conv4] * output_modalities
    output_list = [None] * output_modalities
    layers_list = [{} for x in xrange(output_modalities)]
    previous_layers_list = [{} for x in xrange(output_modalities)]

    for modality in xrange(output_modalities):

        if modality == 0:
            previous_layers_list[modality] = {'conv1': conv1, 'conv2':conv2, 'conv3':conv3}
        else:
            previous_layers_list[modality] = {'conv1': layers_list[modality-1]['conv7'], 'conv2':layers_list[modality-1]['conv6'], 'conv3':layers_list[modality-1]['conv5']}

        layers_list[modality]['up5'] = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2, nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(conv4)
        layers_list[modality]['up5'] = concatenate([layers_list[modality]['up5'], previous_layers_list[modality]['conv3']], axis=1)
        layers_list[modality]['conv5'] = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu', data_format='channels_first',padding='same')(layers_list[modality]['up5'])
        layers_list[modality]['conv5'] = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                       padding='same')(layers_list[modality]['conv5'])

        layers_list[modality]['up6'] = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                         nb_filters=int(256/downsize_filters_factor),image_shape=input_shape[-3:])(layers_list[modality]['conv5'])
        layers_list[modality]['up6'] = concatenate([layers_list[modality]['up6'], previous_layers_list[modality]['conv2']], axis=1)
        layers_list[modality]['conv6'] = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first', padding='same')(layers_list[modality]['up6'])
        layers_list[modality]['conv6'] = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                       padding='same')(layers_list[modality]['conv6'])

        layers_list[modality]['up7'] = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                         nb_filters=int(128/downsize_filters_factor), image_shape=input_shape[-3:])(layers_list[modality]['conv6'])
        layers_list[modality]['up7'] = concatenate([layers_list[modality]['up7'], previous_layers_list[modality]['conv1']], axis=1)
        layers_list[modality]['conv7'] = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first', padding='same')(layers_list[modality]['up7'])
        layers_list[modality]['conv7'] = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',data_format='channels_first',
                       padding='same')(layers_list[modality]['conv7'])

        output_list[modality] = Conv3D(int(1), (1, 1, 1), data_format='channels_first')(layers_list[modality]['conv7'])

    final_output = output_list[0]
    if len(output_list) > 1:
        for output in output_list[1:]:
            final_output = concatenate([final_output, output], axis=1)

    if regression:
        act = Activation('relu')(final_output)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq])
    else:
        act = Activation('sigmoid')(final_output)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def u_net_3d(input_shape=None, input_tensor=None, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, convolutions=4, dropout=.25, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True, output_shape=None):

    # This is messy, as is the part at the conclusion.
    if input_tensor is None:
        inputs = Input(input_shape)
    else:
        inputs = input_tensor

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

    up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2, nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(conv4)
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
                   padding='same')(conv7)

    conv8 = Conv3D(int(num_outputs), (1, 1, 1), data_format='channels_first',)(conv7)

    # Messy
    if input_tensor is not None:
        return conv8

    if regression:
        act = Activation('relu')(conv8)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq])
    else:
        act = Activation('sigmoid')(conv8)
        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def linear_net(input_shape, downsize_filters_factor=1, pool_size=(2, 2, 2), initial_learning_rate=0.00001, convolutions=4, dropout=.25, filter_shape=(3,3,3), num_outputs=1, deconvolution=True, regression=True):

    inputs = Input((1,32,32,32,4))
    conv_mid = Conv3D(int(32/downsize_filters_factor), filter_shape, activation='relu', padding='same', data_format='channels_first')(inputs)

    conv_mid = Dropout(0.25)(conv_mid)

    for conv_num in xrange(convolutions-2):

        conv_mid = Conv3D(int(32/downsize_filters_factor), filter_shape, activation='relu', padding='same', data_format='channels_first')(conv_mid)
        conv_mid = Dropout(0.25)(conv_mid)

    conv_out = Conv3D(int(1), filter_shape, activation='tanh', padding='same', data_format='channels_first', kernel_regularizer=regularizers.l2(0.01))(conv_mid)

    model = Model(inputs=inputs, outputs=conv_out)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=msq_loss, metrics=[msq])

def msq(y_true, y_pred):
    return K.sum(K.pow(y_true - y_pred, 2), axis=None)

def msq_loss(y_true, y_pred):
    return msq(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def load_old_model(model_file):
    print("Loading pre-trained model")

    # custom_objects = {'msq': msq, 'msq_loss': msq_loss}
    custom_objects = {'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef, 'msq': msq, 'msq_loss': msq_loss}

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