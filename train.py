import tables
import os
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from functools import partial
from shutil import rmtree

from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler

from config_files.brats_tumor_config import config

from model import regression_model_3d, load_old_model
from load_data import DataCollection
from data_generator import get_data_generator
from data_utils import pickle_dump, pickle_load
from predict import run_validation_case, run_validation_case_patches
from augment import *

def run_regression(overwrite=False, delete=False, config_dict={}, only_predict=False):

    create_directories(delete=delete)

    modality_dict = config['modality_dict']

    validation_files = []

    # Load training and validation data.
    if config['overwrite_trainval_data'] or not os.path.exists(os.path.abspath(config["hdf5_train"])):

        # Find Data
        validation_data_collection = DataCollection(config['validation_dir'], modality_dict)
        validation_data_collection.fill_data_groups()

        training_data_collection = DataCollection(config['train_dir'], modality_dict)
        training_data_collection.fill_data_groups()

        # Training
        patch_extraction_augmentation = ExtractPatches(config['patch_shape'], config['patch_extraction_conditions'])
        patch_augmentation = AugmentationGroup({'input_modalities': patch_extraction_augmentation, 'ground_truth': patch_extraction_augmentation}, multiplier=20)
        training_data_collection.append_augmentation(patch_augmentation)

        training_data_collection.write_data_to_file(output_filepath = config['hdf5_train'])

        # Validation
        patch_extraction_augmentation = ExtractPatches(config['patch_shape'], config['patch_extraction_conditions'])
        patch_augmentation = AugmentationGroup({'input_modalities': patch_extraction_augmentation, 'ground_truth': patch_extraction_augmentation}, multiplier=5)
        validation_data_collection.append_augmentation(patch_augmentation)

        validation_data_collection.write_data_to_file(output_filepath = config['hdf5_validation'])

    # Create a new model if necessary. Preferably, load an existing one.
    if not config["overwrite_model"] and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        model = regression_model_3d((len(modality_dict['input_modalities']),) + config['patch_shape'], downsize_filters_factor=config['downsize_filters_factor'], initial_learning_rate=config['initial_learning_rate'], regression=config['regression'])

    # Create data generators and train the model.
    if config["overwrite_training"]:
        
        # Get training and validation generators, either split randomly from the training data or from separate hdf5 files.
        if os.path.exists(os.path.abspath(config["hdf5_validation"])):
            open_validation_hdf5 = tables.open_file(config["hdf5_validation"], "r")
            validation_generator, num_validation_steps = get_data_generator(open_validation_hdf5, batch_size=1, data_labels = ['input_modalities', 'ground_truth'])
        else:
            open_validation_hdf5 = []

        open_train_hdf5 = tables.open_file(config["hdf5_train"], "r")
        train_generator, num_train_steps = get_data_generator(open_train_hdf5, batch_size=config["batch_size"], data_labels = ['input_modalities', 'ground_truth'])

        print num_validation_steps, num_train_steps

        # Train model.. TODO account for no validation
        train_model(model=model, model_file=config["model_file"], training_generator=train_generator, validation_generator=validation_generator, steps_per_epoch=num_train_steps, validation_steps=num_validation_steps, initial_learning_rate=config["initial_learning_rate"], learning_rate_drop=config["learning_rate_drop"], learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"])

        # Close training and validation files, no longer needed.
        open_train_hdf5.close()
        if validation_files:
            open_validation_hdf5.close()

    # Load testing data
    if config['overwrite_test_data'] or not os.path.exists(os.path.abspath(config["hdf5_test"])):
        

        testing_data_collection = DataCollection(config['test_dir'], modality_dict)
        testing_data_collection.fill_data_groups()
        testing_data_collection.write_data_to_file(output_filepath = config['hdf5_test'])


    # Run prediction step.
    if config['overwrite_prediction']:
        open_test_hdf5 = tables.open_file(config["hdf5_test"], "r")

        # run_validation_case(output_dir=config['predictions_folder'], model_file=config['model_file'], data_file=open_test_hdf5)
        run_validation_case_patches(output_dir=config['predictions_folder'], model_object=model, model_file=config['model_file'], data_file=open_test_hdf5, patch_shape=config['patch_shape'])

def create_directories(delete=False):

        # Create required directories
    for directory in [config['model_file'], config['hdf5_train'], config['hdf5_test'], config['hdf5_validation'], config['predictions_folder']]:
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            directory = os.path.dirname(directory)
        if delete:
            rmtree(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

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