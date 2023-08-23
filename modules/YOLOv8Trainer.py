import argparse
import os
import loguru
import keras_cv
import tensorflow as tf
from tensorflow import keras
import yaml
from modules.utils import StaticDotDict


class YOLOv8Trainer:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.arg_parse()

    def arg_parse(self):
        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)
        config = StaticDotDict(config)
        return config

    def configure_model(self):
        backbone = keras_cv.models.YOLOV8Backbone.from_preset(
            self.config.model.backbone
        )
        yolo_v8_model = keras_cv.models.YOLOV8Detector(
            backbone=backbone,
            num_classes=len(self.config.model.classes),
            bounding_box_format=self.config.model.bounding_box_format,
            fpn_depth=self.config.model.fpn_depth,
        )
        self.model = yolo_v8_model

    def train():
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def configure_callback(self):
        raise NotImplementedError
