import tables

from config_files.dummy_config import config

from model import regression_model_3d
from load_data import fetch_data_files, write_data_to_file
from data_generator import get_training_and_validation_generators

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
    train_generator, validation_generator, nb_train_samples, nb_test_samples = get_training_and_validation_generators(
        hdf5_file_opened, batch_size=config["batch_size"], data_split=config["validation_split"], overwrite=overwrite,
        validation_keys_file=config["validation_file"], training_keys_file=config["training_file"],
        n_labels=config["n_labels"])

    # run training
    train_model(model=model, model_file=config["model_file"], training_generator=train_generator, validation_generator=validation_generator, steps_per_epoch=nb_train_samples, validation_steps=nb_test_samples, initial_learning_rate=config["initial_learning_rate"], learning_rate_drop=config["learning_rate_drop"], learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"])
    hdf5_file_opened.close()


def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps, initial_learning_rate, learning_rate_drop, learning_rate_epochs, n_epochs):
    """
    Train a Keras model.
    :param model: Keras model that will be trained. 
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """
    model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs, validation_data=validation_generator, validation_steps=validation_steps, pickle_safe=True, callbacks=get_callbacks(model_file, initial_learning_rate=initial_learning_rate, learning_rate_drop=learning_rate_drop,learning_rate_epochs=learning_rate_epochs))

    model.save(model_file)

if __name__ == '__main__':
    run_regression(overwrite=True)