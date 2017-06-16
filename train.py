import tables
import os
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from functools import partial
from shutil import rmtree

from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler

from config_files.resample_config import config

from model import regression_model_3d, load_old_model
from load_data import fetch_data_files, write_data_to_file
from data_generator import get_training_and_validation_generators
from data_utils import pickle_dump, pickle_load
from predict import run_validation_case, run_validation_case_patches

def run_regression(overwrite=False, delete=False, config_dict={}, only_predict=False):

    validation_files = []

    # Create required directories
    for directory in [config['model_file'], config['hdf5_train'], config['hdf5_test'], config['hdf5_validation'], config['predictions_folder']]:
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            directory = os.path.dirname(directory)
        if delete:
            rmtree(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Load training and validation data.
    if config['overwrite_trainval_data'] or not os.path.exists(os.path.abspath(config["hdf5_train"])):
        training_data_files, training_groundtruth_files = fetch_data_files(config['train_dir'], config['input_modalities'], config['input_groundtruth'])
        write_data_to_file(training_data_files, training_groundtruth_files, config['hdf5_train'], config['image_shape'], patches=config['patches'], patch_num=config['train_patch_num'], patch_shape=config['patch_shape'], patch_extraction_conditions=config['patch_extraction_conditions'])

    # Optionally, load validation data.
    if config['overwrite_trainval_data'] or not os.path.exists(os.path.abspath(config["hdf5_validation"])):
        if config['validation_dir'] != '':
            validation_data_files, validation_groundtruth_files = fetch_data_files(config['validation_dir'], config['input_modalities'], config['input_groundtruth'])
            write_data_to_file(validation_data_files, validation_groundtruth_files, config['hdf5_validation'], config['image_shape'], patches=config['patches'], patch_num=config['validation_patch_num'], patch_shape=config['patch_shape'], patch_extraction_conditions=config['patch_extraction_conditions'])
        else:
            validation_files = []

    # Open up all relevant hdf5 files..
    if os.path.exists(os.path.abspath(config["hdf5_validation"])):
        open_validation_hdf5 = tables.open_file(config["hdf5_validation"], "r")
    else:
        open_validation_hdf5 = []

    if config["overwrite_training"]:
        open_train_hdf5 = tables.open_file(config["hdf5_train"], "r")
        # Get training and validation generators, either split randomly from the training data or from separate hdf5 files.
        train_generator, validation_generator, num_train_steps, num_test_steps = get_training_and_validation_generators(training_data_file=open_train_hdf5, validation_data_file=open_validation_hdf5, batch_size=config["batch_size"], train_test_split=config["train_test_split"], overwrite=config["overwrite_train_val_split"], validation_keys_file=config["validation_file"], training_keys_file=config["training_file"])

    # Create a new model if necessary. Preferably, load an existing one.
    if not config["overwrite_model"] and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        model = regression_model_3d((len(config['input_modalities']),) + config['patch_shape'], downsize_filters_factor=config['downsize_filters_factor'], initial_learning_rate=config['initial_learning_rate'], regression=config['regression'])

    # Train the model!
    if config["overwrite_training"]:
        
        train_model(model=model, model_file=config["model_file"], training_generator=train_generator, validation_generator=validation_generator, steps_per_epoch=num_train_steps, validation_steps=num_test_steps, initial_learning_rate=config["initial_learning_rate"], learning_rate_drop=config["learning_rate_drop"], learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"])

        # Close training and validation files, no longer needed.
        open_train_hdf5.close()
        if validation_files:
            open_validation_hdf5.close()

    # Load testing data
    if config['overwrite_test_data'] or not os.path.exists(os.path.abspath(config["hdf5_test"])):
        testing_data_files, testing_groundtruth_files = fetch_data_files(config['test_dir'], config['input_modalities'], config['input_groundtruth'])
        write_data_to_file(testing_data_files, testing_groundtruth_files, config["hdf5_test"], config['image_shape'])  

    open_test_hdf5 = tables.open_file(config["hdf5_test"], "r")

    # run_validation_case(output_dir=config['predictions_folder'], model_file=config['model_file'], data_file=open_test_hdf5)
    run_validation_case_patches(output_dir=config['predictions_folder'], model_object=model, model_file=config['model_file'], data_file=open_test_hdf5, patch_shape=config['patch_shape'])

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
    run_regression(overwrite=False, only_predict=False)