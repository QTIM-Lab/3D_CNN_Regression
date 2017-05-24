import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation
from keras.optimizers import Adam

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

def regression_model_3d(input_shape, downsize_filters_factor=1, pool_size=(2, 2, 2), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):

    # 144x144x144
    inputs = Input(input_shape)

    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(inputs)

    conv1 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv1)

    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv1)

    conv1 = Conv3D(int(1/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv1)

    model = Model(inputs=inputs, outputs=conv1)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


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
    return tuple([None, filters] + [int(x) for x in output_image_shape])


def get_upconv(depth, nb_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2),
               deconvolution=False):
    if deconvolution:
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
        return UpSampling3D(size=pool_size)

if __name__ == '__main__':

  # config["pool_size"] = (2, 2, 2)
  # config["image_shape"] = (144, 144, 144)  # This determines what shape the images will be cropped/resampled to.
  # config["n_labels"] = 1  # not including background

  print regression_model_3d((4, 144, 144, 144), downsize_filters_factor=1, pool_size=(2, 2, 2), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False)