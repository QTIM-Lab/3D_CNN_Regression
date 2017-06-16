import os

config = dict()

# Data will be compressed in hdf5 format at these filepaths.
config["train_dir"] = os.path.abspath("~/Train")
config["test_dir"] = os.path.abspath("~/Test")
# config["test_dir"] = os.path.abspath("~/TATA")
config["validation_dir"] = os.path.abspath("~/Validation")

# Data will be saved to these hdf5 files.
config["hdf5_train"] = './hdf5_data/brats_train.hdf5'
config["hdf5_test"] = './hdf5_data/brats_test.hdf5'
config["hdf5_validation"] = './hdf5_data/brats_validation.hdf5'

# If you want to preserve automated training/validation splits..
config["training_file"] = os.path.abspath("./hdf5_data/training_ids.pkl")
config["validation_file"] = os.path.abspath("./hdf5_data/validation_ids.pkl")

# Overwrite settings.
config["overwrite_trainval_data"] = True
config["overwrite_train_val_split"] = True
config['overwrite_test_data'] = True
config["overwrite_model"] = True
config["overwrite_training"] = True

# Image Information
config["image_shape"] = (240, 240, 155)

# Patch Information
config['patches'] = True
config['patch_shape'] = (16, 16, 16)
config['train_patch_num'] = 6000
config['validation_patch_num'] = 3000

# Modalities. Always make input_groundtruth as list.
config["input_modalities"] = ['FLAIR_pp', 'T2_pp', 'T1c_pp', 'T1_pp']
config["input_groundtruth"] = ['ROI_pp_edema_nonenhancing_tumor_necrosis']
# config["input_modalities"] = ['FLAIR_p', 'T2_p', 'T1C_p', 'T1_p']
# config["input_groundtruth"] = ['GT_p']
config["regression"] = True

# Path to save model.
config["model_file"] = "./model_files/brats_model_regress.h5"

# Model parameters
config["downsize_filters_factor"] = 1
config["decay_learning_rate_every_x_epochs"] = 20
config["initial_learning_rate"] = 0.0001
config["learning_rate_drop"] = 0.9
config["n_epochs"] = 600

# Model training parameters
config["train_test_split"] = .8
config["batch_size"] = 50

# Model testing parameters
config['predictions_folder'] = os.path.abspath('./predictions')

# Threshold Functions
def background_patch(patch):
    return float((patch[0,...] == 0).sum()) / patch[0,...].size == 1
def brain_patch(patch):
    return float((patch[0,...] > 0).sum()) / patch[0,...].size > .5 and float((patch[-1,...] == 1).sum()) / patch[0,...].size < .5
def roi_patch(patch):
    return float((patch[-1,...] == 1).sum()) / patch[-1,...].size > .5

config["patch_extraction_conditions"] = [[background_patch, .001], [brain_patch, .199], [roi_patch, .8]]

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
