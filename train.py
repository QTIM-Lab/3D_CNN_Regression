import tables
import os
import math
from functools import partial

from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler

from config_files.dummy_config import config

from model import regression_model_3d, load_old_model
from load_data import fetch_data_files, write_data_to_file
from data_generator import get_training_and_validation_generators
from data_utils import pickle_dump, pickle_load
from predict import run_validation_case

def run_regression(overwrite=False):

    input_image_shape = config['image_shape']

    # Load training data.
    if overwrite or not os.path.exists(os.path.abspath(config["hdf5_train"])):
        training_files = fetch_data_files(config['train_dir'], config['input_modalities'], config['input_groundtruth'])
        write_data_to_file(training_files, config["hdf5_train"], input_image_shape)

    # Load tresting data. We could combine them
    # into one hdf5, but I feel that this provides more
    # flexibility for testing data kept in a different
    # place
    if overwrite or not os.path.exists(os.path.abspath(config["hdf5_test"])):
        testing_files = fetch_data_files(config['test_dir'], config['input_modalities'], config['input_groundtruth'])
        write_data_to_file(testing_files, config["hdf5_test"], input_image_shape)  

    open_train_hdf5, open_test_hdf5 = tables.open_file(config["hdf5_train"], "r"), tables.open_file(config["hdf5_test"], "r")

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        print (len(config['input_modalities']),)
        print config['image_shape']
        model = regression_model_3d((len(config['input_modalities']),) + config['image_shape'], downsize_filters_factor=config['downsize_filters_factor'], initial_learning_rate=config['initial_learning_rate'])

    # get training and testing generators
    train_generator, validation_generator, num_train_steps, num_test_steps = get_training_and_validation_generators(open_train_hdf5, batch_size=config["batch_size"], train_test_split=config["train_test_split"], overwrite=overwrite, validation_keys_file=config["validation_file"], training_keys_file=config["training_file"])

    print train_generator, validation_generator, num_train_steps, num_test_steps

    # run training
    train_model(model=model, model_file=config["model_file"], training_generator=train_generator, validation_generator=validation_generator, steps_per_epoch=num_train_steps, validation_steps=num_test_steps, initial_learning_rate=config["initial_learning_rate"], learning_rate_drop=config["learning_rate_drop"], learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"])

    # Close model.
    open_train_hdf5.close()

    run_validation_case(output_dir=config['predictions_folder'], model_file=config['model_file'], data_file=open_test_hdf5)

def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps, initial_learning_rate, learning_rate_drop, learning_rate_epochs, n_epochs):

    model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs, validation_data=validation_generator, validation_steps=validation_steps, pickle_safe=True, callbacks=get_callbacks(model_file, initial_learning_rate=initial_learning_rate, learning_rate_drop=learning_rate_drop,learning_rate_epochs=learning_rate_epochs))

    model.save(model_file)

""" The following three functions/classes are mysterious to me.
"""

def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

class SaveLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        pickle_dump(self.losses, "loss_history.pkl")

def get_callbacks(model_file, initial_learning_rate, learning_rate_drop, learning_rate_epochs, logging_dir="."):

    """ Currently do not understand callbacks.
    """

    model_checkpoint = ModelCheckpoint(model_file, save_best_only=True)
    logger = CSVLogger(os.path.join(logging_dir, "training.log"))
    history = SaveLossHistory()
    scheduler = LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate, drop=learning_rate_drop, epochs_drop=learning_rate_epochs))
    return [model_checkpoint, logger, history, scheduler]


if __name__ == '__main__':
    run_regression(overwrite=True)