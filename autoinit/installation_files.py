import os
import zipfile
from pathlib import Path

DRIVE_ROOT_DIR = "/content/gdrive/MyDrive/pysource_object_detection/"
DARKNET_PATH = "/content/darknet"

def is_gpu_enabled():
    # Check if GPU is enabled
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))



# Connect google drive
def connect_google_drive(project_name):
    from google.colab import drive
    drive.mount('/content/gdrive')

    model_dir = os.path.join(DRIVE_ROOT_DIR, project_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("New project created {}".format(project_name))
        print("You'll find the project on Google Drive, on the folder pysource_object_detection/{} .".format(project_name))
    else:
        print("Project {} already exists. Editing existing project.".format(project_name))
    return model_dir

def unzip_dataset(project_name):
    dataset_path = os.path.join(DRIVE_ROOT_DIR, project_name)
    dataset_path = os.path.join(dataset_path, "dataset.zip")
    output_path = "/content/darknet/data/obj"
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall("/content/darknet/data/obj")


def extract_dataset(project_name):
    print("Extracting dataset ...")
    dataset_path = os.path.join(DRIVE_ROOT_DIR, project_name)
    dataset_path = os.path.join(dataset_path, "dataset.zip")

    print("Dataset path: {}".format(dataset_path))

    # Dataset exists
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError("{} doesn't exist, make sure you uploaded the file on the correct folder and that its name is dataset.zip.")

    output_path = Path("/content/darknet/data/obj")

    with zipfile.ZipFile(dataset_path) as zip:
        for zip_info in zip.infolist():
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            zip.extract(zip_info, output_path)

    print("Dataset Extracted")


def find_existing_weights(project_name):
    dnn_path = "/content/gdrive/MyDrive/pysource_object_detection/{}/dnn".format(project_name)
    new_weights_path = os.path.join(dnn_path, "custom-detector_last.weights")
    # check if new weights exists
    if os.path.exists(new_weights_path):
        print("Existing weigh file found on: {}".format(new_weights_path))
        print("Resuming interrupted training")
        return new_weights_path