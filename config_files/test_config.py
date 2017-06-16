import os

# from sigopt import Connection
# from sigopt.examples import franke_function

# conn = Connection(client_token="TPTUMZGIDOENQNPIBCFOVEARZOSYKOVZGMYMCJYJOUKPZHXP")

# experiment = conn.experiments().create(
#     name='FMS Optimization',
#     parameters=[
#         dict(name='convolution_layers', type='int', bounds=dict(min=1, max=15)),
#         dict(name='patch_shape', type='int', bounds=dict(min=2, max=32)),
#         dict(name='train_patch_num', type='int', bound=dict(min=10, 5000)),
#         dict(name='test_patch_num', type='int', bound=dict(min=10, 5000)),
#         dict(name='filter', type='int', bound=dict(min=10, 5000)),
#     ],
# )

# print("Created experiment: https://sigopt.com/experiment/" + experiment.id)

# # Evaluate your model with the suggested parameter assignments
# # Franke function - http://www.sfu.ca/~ssurjano/franke2d.html

# # Run the Optimization Loop between 10x - 20x the number of parameters
# for _ in range(30):
#     suggestion = conn.experiments(experiment.id).suggestions().create()
#     value = evaluate_model(suggestion.assignments)
#     conn.experiments(experiment.id).observations().create(
#         suggestion=suggestion.id,
#         value=value,
#     )


config = dict()

# Data will be compressed in hdf5 format at these filepaths.
config["train_dir"] = 
config["test_dir"] = 
config["validation_dir"] = 

# Data will be saved to these hdf5 files.
config["hdf5_train"] = './hdf5_data/FMS_train.hdf5'
config["hdf5_test"] = './hdf5_data/FMS_test.hdf5'
config["hdf5_validation"] = './hdf5_data/FMS_validation.hdf5'

# If you want to preserve automated training/validation splits..
config["training_file"] = os.path.abspath("./hdf5_data/training_ids.pkl")
config["validation_file"] = os.path.abspath("./hdf5_data/validation_ids.pkl")

# Image Information
config["image_shape"] = (256, 256, 208)

# Patch Information
config['patches'] = True
config['patch_shape'] = (16, 16, 16)
config['train_patch_num'] = 600
config['validation_patch_num'] = 300

# Modalities. Always make input_groundtruth as list.
config["input_modalities"] = ['MPRAGE_POST', 'FLAIR_r_T2', 'T2SPACE_DL', 'T1Pre']
config["input_groundtruth"] = ['SUV']

# Path to save model.
config["model_file"] = 

# Model parameters
config["downsize_filters_factor"] = 1
config["decay_learning_rate_every_x_epochs"] = 10
config["initial_learning_rate"] = 0.001
config["learning_rate_drop"] = 0.9
config["n_epochs"] = 200

# Model training parameters
config["train_test_split"] = .8
config["batch_size"] = 50

# Model testing parameters
config['predictions_folder'] = os.path.abspath('./predictions')

def evaluate_model(assignments):
    train(config)

# config["pool_size"] = (2, 2, 2)  # This determines what shape the images will be cropped/resampled to.
# config["n_labels"] = 1  # not including background
# config["n_epochs"] = 50
# config["decay_learning_rate_every_x_epochs"] = 10
# config["initial_learning_rate"] = 0.00001
# config["learning_rate_drop"] = 0.5
# config["validation_split"] = 0.8
# config["smooth"] = 1.0
# # config["nb_channels"] = len(config["training_modalities"])
# # config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
# config["truth_channel"] = config["nb_channels"]
# config["background_channel"] = config["nb_channels"] + 1
# config["deconvolution"] = False  # use deconvolution instead of up-sampling. Requires keras-contrib.
# # divide the number of filters used by by a given factor. This will reduce memory consumption.
