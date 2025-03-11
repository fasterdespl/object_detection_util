from mrcnn.mask_rcnn_pro_engine import CustomConfig, train_head, \
    train_all_layers, load_image_dataset, load_training_model, load_test_model, \
    test_random_image, model_evaluation
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()


parser.add_argument('-f', "--function")
parser.add_argument('-n', "--project_name", type=str)
parser.add_argument('-e', "--epochs", type=int)
parser.add_argument('-s', "--steps_per_epoch", type=int)
parser.add_argument('-a', "--augmentation", type=bool)
parser.add_argument('-i', "--image_size", type=int)
parser.add_argument('-c', "--continue_training", type=bool, default=False)
# add argument


args = parser.parse_args()
args_var = vars(args)

# extract variables from args

PROJECT_NAME = args_var['project_name']
EPOCHS = args_var['epochs']
STEPS_PER_EPOCH = args_var['steps_per_epoch']
AUGMENTATION = args_var['augmentation']
IMAGE_SIZE = args_var['image_size']
CONTINUE_TRAINING = args_var['continue_training']



def train():
    global PROJECT_NAME, EPOCHS, STEPS_PER_EPOCH, AUGMENTATION, IMAGE_SIZE, CONTINUE_TRAINING
    # Load Configuration
    dataset_train, dataset_val, class_number = load_image_dataset(PROJECT_NAME)
    config = CustomConfig(class_number, STEPS_PER_EPOCH, IMAGE_SIZE)
    # config.display()
    # If continue training load model with last epoch
    if CONTINUE_TRAINING:
        model = load_training_model(config, PROJECT_NAME, init_with="last")
    else:
        model = load_training_model(config, PROJECT_NAME)
    train_head(model, dataset_train, dataset_val, config, EPOCHS, AUGMENTATION)

def train_fine_tuning():
    global PROJECT_NAME, EPOCHS, STEPS_PER_EPOCH, AUGMENTATION, IMAGE_SIZE, CONTINUE_TRAINING
    dataset_train, dataset_val, class_number = load_image_dataset(PROJECT_NAME)
    config = CustomConfig(class_number, STEPS_PER_EPOCH, IMAGE_SIZE)
    model = load_training_model(config, PROJECT_NAME, init_with="last")
    train_all_layers(model, dataset_train, dataset_val, config, EPOCHS, AUGMENTATION)

def test_model():
    global PROJECT_NAME, EPOCHS, STEPS_PER_EPOCH, AUGMENTATION, IMAGE_SIZE
    dataset_train, dataset_val, class_number = load_image_dataset(PROJECT_NAME)
    test_model, inference_config = load_test_model(class_number, PROJECT_NAME)
    test_random_image(test_model, dataset_val, inference_config)

def evaluation():
    global PROJECT_NAME, EPOCHS, STEPS_PER_EPOCH, AUGMENTATION, IMAGE_SIZE
    # The evaluation of the Model is performed on the Validation images
    dataset_train, dataset_val, class_number = load_image_dataset(PROJECT_NAME)
    test_model, inference_config = load_test_model(class_number, PROJECT_NAME)
    APs = model_evaluation(dataset_val, test_model, inference_config)
    print("mAP: ", np.mean(APs))


FUNCTION_MAP = {'train' : train,
                'test_model' : test_model,
                'train_fine_tuning' : train_fine_tuning,
                'evaluation' : evaluation}

func = FUNCTION_MAP[args.function]
func()