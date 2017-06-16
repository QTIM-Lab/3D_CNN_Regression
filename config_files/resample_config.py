import os

config = dict()

# Data will be compressed in hdf5 format at these filepaths.
config["train_dir"] = 
# config["test_dir"] = 
config["test_dir"] = 
config["validation_dir"] = 

# Data will be saved to these hdf5 files.
config["hdf5_train"] = './hdf5_data/downsample_train.hdf5'
config["hdf5_test"] = './hdf5_data/downsample_test.hdf5'
config["hdf5_validation"] = './hdf5_data/downsample_validation.hdf5'

# If you want to preserve automated training/validation splits..
config["training_file"] = os.path.abspath("./hdf5_data/training_ids.pkl")
config["validation_file"] = os.path.abspath("./hdf5_data/validation_ids.pkl")

# Overwrite settings.
config["overwrite_trainval_data"] = False
config["overwrite_train_val_split"] = False
config["overwrite_model"] = False
config["overwrite_training"] = False
config["overwrite_test_data"] = False

# Image Information
# config["image_shape"] = (240, 240, 155)
config["image_shape"] = (256, 256, 208)

# Patch Information
config['patches'] = True
config['patch_shape'] = (32, 32, 32)
config['train_patch_num'] = 1600
config['validation_patch_num'] = 600

# Modalities. Always make input_groundtruth as list.
# config["input_modalities"] = ['FLAIR_pp_ds_us', 'T2_pp.', 'T1c_pp_ds_us', 'T1_pp_ds_us', 'ROI_ds_us']
# config["input_groundtruth"] = ['ROI.']
config["input_modalities"] = ['FLAIR_r', 'T2SP', 'T1Post', 'T1Pre', 'UPSAMPLED']
config["input_groundtruth"] = ['UPSAMPLED']

# Path to save model.
config["model_file"] = "./model_files/downsample_model_regress.h5"

# Model Structure
config["regression"] = True
config["num_outputs"] = 4

# Model parameters
config["downsize_filters_factor"] = 2
config["decay_learning_rate_every_x_epochs"] = 20
config["initial_learning_rate"] = 0.0001
config["learning_rate_drop"] = 0.9
config["n_epochs"] = 30

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
    return float((patch[-1,...] >= 1).sum()) / patch[-1,...].size > .1

config["patch_extraction_conditions"] = [[roi_patch, 1]]

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
