# This files prepare a custom Object detector for YOLO v4
#
import glob
import errno
import os
import argparse
import re
import urllib.request
import zipfile
from pathlib import Path

DARKNET_PATH = "/content/darknet"
#
resume_interrupted = False


# ./darknet partial cfg/yolov3-tiny-relu-tpu.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15

class CustomYOLODetector:
    def __init__(self, ProjectName, Model):
        # Files location
        self.cfg_paths = {"yolov4": "cfg/yolov4-custom.cfg",
                     "yolov4-tiny": "cfg/yolov4-tiny-custom.cfg",
                     "yolov4-csp": "cfg/yolov4-csp.cfg",
                     "yolov4-tiny-relu-tpu" : "cfg/yolov4-tiny-relu-tpu.cfg",
                     "yolov3-tiny-relu-tpu" : "cfg/yolov3-tiny-relu-tpu.cfg"}

        self.subdivisions_by_model = {"yolov4": 32,
                                      "yolov4-tiny": 8,
                                      "yolov4-csp": 32,
                                      "yolov4-tiny-relu-tpu": 8,
                                      "yolov3-tiny-relu-tpu": 8,}
        

        self.weights_by_model = {"yolov4": "yolov4.conv.137",
                            "yolov4-tiny": "yolov4-tiny.conv.29",                         
                            "yolov4-csp": "yolov4-csp.conv.142",
                            "yolov4-tiny-relu-tpu": "yolov4-tiny.conv.29",
                            "yolov3-tiny-relu-tpu": "yolov3-tiny.conv.15"}


        self.projectname = ProjectName
        self.model = Model

        self.drive_folder = r"/content/gdrive/MyDrive/projects_object_detection"
        self.custom_cfg_path = self.cfg_paths.get(self.model)
        self.new_custom_cfg_path = "cfg/{}-custom.cfg".format(self.model)
        self.dnn_path = "{}/{}/dnn".format(self.drive_folder, self.projectname)
        self.new_custom_cfg_test_path = "{}/{}/dnn/{}-custom.cfg".format(self.drive_folder, self.projectname, self.model)
        self.obj_data_path = "data/obj.data"
        self.obj_names_path = "data/obj.names"
        self.images_folder_path = "data/obj/"
        self.backup_folder_path = "{}/{}/dnn/".format(self.drive_folder, self.projectname)

        self.n_classes = 0
        self.n_labels = 0
        
        self.test_images_list = []

        print("YOLO MODEL: {}".format(self.model))
        
    def download_dnn_model(self):
        # change makefile to have GPU and OPENCV enabled

        # Download Weights
        print("Downloading model weights: {}".format(self.model))
        if self.model == "yolov4-csp":
            urllib.request.urlretrieve(
                'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.conv.142',
                os.path.join(DARKNET_PATH, 'yolov4-csp.conv.142'))
        elif (self.model == "yolov4-tiny") or (self.model == "yolov4-tiny-relu-tpu"):
            urllib.request.urlretrieve(
                'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29',
                os.path.join(DARKNET_PATH, 'yolov4-tiny.conv.29'))
        elif self.model == "yolov4":
            urllib.request.urlretrieve(
                'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137',
                os.path.join(DARKNET_PATH, 'yolov4.conv.137'))
        elif self.model == "yolov3-tiny-relu-tpu":
            urllib.request.urlretrieve(
                'https://github.com//hamzaMahdi/darknet/raw/master/yolov3-tiny.conv.15',
                os.path.join(DARKNET_PATH, 'yolov3-tiny.conv.15'))    

        print("Weights downloaded")

    def count_classes_number(self):
        print("Detecting classes number ...")
        # Detect number of Classes by reading the labels indexes
        # If there are missing indexes, normalize the number of classes by rewriting the indexes starting from 0
        txt_file_paths = glob.glob(self.images_folder_path + "**/*.txt", recursive=True)
        self.n_labels = len(txt_file_paths)
        print("{} label files found".format(self.n_labels))
        # Count number of classes
        class_indexes = set()
        for i, file_path in enumerate(txt_file_paths):
            # get image size
            with open(file_path, "r") as f_o:
                lines = f_o.readlines()
                for line in lines:
                    numbers = re.findall("[0-9.]+", line)
                    if numbers:
                        # Define coordinates
                        class_idx = int(numbers[0])
                        class_indexes.add(class_idx)

        # Update classes number
        self.n_classes = len(class_indexes)
        print("{} classes found".format(self.n_classes))

        # Verify if there are missing indexes
        if max(class_indexes) > len(class_indexes) - 1:
            print("Class indexes missing")
            print("Normalizing and rewriting classes indexes so that they have consecutive index number")
            # Assign consecutive indexes, if there are missing ones
            # for example if labels are 0, 1, 3, 4 so the index 2 is missing
            # rewrite labels with indexes 0, 1, 2, 3
            new_indexes = {}
            classes = sorted(class_indexes)
            for i in range(len(classes)):
                new_indexes[classes[i]] = i

            for i, file_path in enumerate(txt_file_paths):
                # get image size
                with open(file_path, "r") as f_o:
                    lines = f_o.readlines()
                    text_converted = []
                    for line in lines:
                        numbers = re.findall("[0-9.]+", line)
                        if numbers:
                            # Define coordinates
                            class_idx = new_indexes.get(int(numbers[0]))
                            class_indexes.add(class_idx)
                            text = "{} {} {} {} {}".format(0, numbers[1], numbers[2], numbers[3], numbers[4])
                            text_converted.append(text)
                    # Write file
                    with open(file_path, 'w') as fp:
                        for item in text_converted:
                            fp.writelines("%s\n" % item)

    def generate_yolo_custom_cfg(self, flag="training"):
        """
        This files loads the yolo
        :return:
        """

        # 1) Edit CFG file
        if flag == "training":
            print("Generating .cfg file for {} classes".format(self.n_classes))
        # print("Classes number: {}")

        # Edit subdivision, max_batches, classes and filters on configuration file
        # Max batches 2000*class, but a minimum of 6000
        max_batches = self.n_classes * 2000
        max_batches = 6000 if max_batches < 6000 else max_batches
        max_batches = self.n_labels if max_batches < self.n_labels else max_batches
        if flag == "training":
            print("Settings:")
            print("max batches: {}".format(max_batches))

        # Choose subdivisions
        subdivisions = self.subdivisions_by_model.get(self.model)

        if flag == "test":
            batch_size = 1
            subdivisions = 1

        with open(self.custom_cfg_path, "r") as f_o:
            cfg_lines = f_o.readlines()

        lines = []
        for line in cfg_lines:
            if re.search("subdivisions=[0-9]+\n", line):
                #print("subdivisions: {}".format(args["subdivisions"]))
                new_line = "subdivisions={}\n".format(subdivisions)
                lines.append(new_line)
                continue
            # elif re.search("width=[0-9]+\n", line) and args["imagesize"]:
            #     print("width: {}".format(args["imagesize"]))
            #     new_line = "width={}\n".format(args["imagesize"])
            #     lines.append(new_line)
            #     continue
            # elif re.search("height=[0-9]+\n", line) and args["imagesize"]:
            #     print("height: {}".format(args["imagesize"]))
            #     new_line = "height={}\n".format(args["imagesize"])
            #     lines.append(new_line)
            #     continue
            elif re.search("max_batches[0-9\s=]+\n", line):
                new_line = "max_batches = {}\n".format(max_batches)
                lines.append(new_line)
                continue
            elif re.search("steps[0-9\s=,]+\n", line):
                new_line = "steps={},{}\n".format(int(max_batches * 0.8), int(max_batches * 0.9))
                lines.append(new_line)
                continue
            elif re.search("classes[0-9\s=]+\n", line):
                new_line = "classes={}\n".format(self.n_classes)
                lines.append(new_line)
                continue
            elif re.search("filters=255\n", line):
                filters = (self.n_classes + 5) * 3
                new_line = "filters={}\n".format(filters)
                lines.append(new_line)
                continue

            # Append the same line if different parameters are not found
            lines.append(line)

        # Saving edited file
        if not os.path.exists(self.dnn_path):
            os.makedirs(self.dnn_path)
        if flag == "training":
            with open(self.new_custom_cfg_path, "w") as f_o:
                f_o.writelines(lines)
        else:
            with open(self.new_custom_cfg_test_path, "w") as f_o:
                f_o.writelines(lines)

    def generate_obj_data(self):
        obj_data = 'classes= {}\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = {}' \
            .format(self.n_classes, self.backup_folder_path)

        # Saving Obj data
        with open(self.obj_data_path, "w") as f_o:
            f_o.writelines(obj_data)

        # Saving Obj names
        with open(self.obj_names_path, "w") as f_o:
            for i in range(self.n_classes):
                f_o.writelines("CLASS {}\n".format(i))

    def generate_train_val_files(self):
        print("Generating Train/Validation list")
        images_list = glob.glob(self.images_folder_path + "**/*.jpg", recursive=True)
        print(images_list)
        print("{} Images found".format(len(images_list)))
        if len(images_list) == 0:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), "Images list not found. Make sure that the images are '.jpg' "
                                                         "format and inside the directory {}".format(
                    self.images_folder_path))

        # Read labels
        for img_path in images_list:
            img_name_basename = os.path.basename(img_path)
            img_name = os.path.splitext(img_name_basename)[0]

        # Generate test files, 10% of training
        test_number = len(images_list) // 10
        print("Test images: {}".format(test_number))
        self.test_images_list = []
        with open("data/test.txt", "w") as f_o:
            for i, path in enumerate(images_list):
                f_o.writelines("{}\n".format(path))
                self.test_images_list.append(path)
                if i == test_number:
                    break
        print("Test.txt generated")
        
        with open("data/train.txt", "w") as f_o:
            for i, path in enumerate(images_list):
                if i > test_number:
                    f_o.writelines("{}\n".format(path))
        print("Train.txt generated")
        

    def find_existing_weights(self):
        new_weights_path = os.path.join(self.dnn_path, "{}-custom_last.weights".format(self.model))
        # check if new weights exists
        if os.path.exists(new_weights_path):
            print("Existing weigh file found on: {}".format(new_weights_path))
            print("Resuming interrupted training")
            return new_weights_path
        else:
            print("Starting new training")
            return self.weights_by_model[self.model]

    def run(self):
        self.count_classes_number()
        self.generate_yolo_custom_cfg()
        self.generate_yolo_custom_cfg("test")
        self.generate_obj_data()
        self.generate_train_val_files()
        self.download_dnn_model()



if "__main__" == __name__:
    cyd = CustomYOLODetector()
