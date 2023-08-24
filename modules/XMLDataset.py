from modules.utils import StaticDotDict
import yaml
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
import os
import tensorflow as tf
from keras_cv import bounding_box


class XMLDataset:
    def __init__(self, config_file):
        self._config_file = config_file
        self.config = self._arg_parse()
        self.path_images = os.path.join(self.config.dataset.path, "images")
        self.path_annotations = os.path.join(self.config.dataset.path, "Annotations")
        self.class_ids = self.config.model.classes
        self.class_mapping = dict(zip(range(len(self.class_ids)), self.class_ids))

    def _arg_parse(self):
        with open(self._config_file, "r") as f:
            config = yaml.safe_load(f)
        config = StaticDotDict(config)
        return config

    def _get_xml_files(self):
        self.xml_files = sorted(
            [
                os.path.join(self.path_annotations, file_name)
                for file_name in os.listdir(self.path_annotations)
                if file_name.endswith(".xml")
            ]
        )

    def _get_jpg_files(self):
        self.jpg_files = sorted(
            [
                os.path.join(self.path_images, file_name)
                for file_name in os.listdir(self.path_images)
                if file_name.endswith(".jpg")
            ]
        )

    def _parse_annotation(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_name = root.find("filename").text
        image_path = os.path.join(self.path_images, image_name)

        boxes = []
        classes = []
        for obj in root.iter("object"):
            cls = obj.find("name").text
            classes.append(cls)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        class_ids = [
            list(self.class_mapping.keys())[
                list(self.class_mapping.values()).index(cls)
            ]
            for cls in classes
        ]
        return image_path, boxes, class_ids

    def _dict_to_tuple(self, inputs):
        return inputs["images"], inputs["bounding_boxes"]

    def _dict_to_tuple_tpu(inputs):
        return inputs["images"], bounding_box.to_dense(
            inputs["bounding_boxes"], max_boxes=32
        )

    def _load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image

    def _load_dataset(self, image_path, classes, bbox):
        image = self._load_image(image_path)
        boxes = bounding_box.convert_format(
            bbox,
            images=image,
            source=self.config.dataset.previous_bounding_box_format,
            target=self.config.model.bounding_box_format,
        )
        bounding_boxes = {
            "classes": tf.cast(classes, dtype=tf.float32),
            "boxes": tf.cast(boxes, dtype=tf.float32),
        }
        return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

    def build_data(self):
        self._get_xml_files()
        self._get_jpg_files()
        image_paths = []
        bbox = []
        classes = []
        for xml_file in tqdm(self.xml_files):
            image_path, boxes, class_ids = self._parse_annotation(xml_file)
            image_paths.append(image_path)
            bbox.append(boxes)
            classes.append(class_ids)
        bbox = tf.ragged.constant(bbox)
        classes = tf.ragged.constant(classes)
        image_paths = tf.ragged.constant(image_paths)
        self.data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

    def build_dataset(self):
        if self.dataset.val_split == 0:
            self.train_ds = self.data.map()
            self.train_ds = self.train_data.map(
                self._load_dataset, num_parallel_calls=tf.data.AUTOTUNE
            )
            self.train_ds = self.train_ds.ragged_batch(
                self.config.batch_size, drop_remainder=True
            )
