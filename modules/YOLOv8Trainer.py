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
        self._config_file = config_file
        self.config = self._arg_parse()

    def _arg_parse(self):
        with open(self._config_file, "r") as f:
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

    def compile(self):
        if self.config.train.accelerator == "tpu":
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                "local"
            )
            tf.config.experimental_connect_to_cluster(cluster_resolver)
            tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
            self.strategy = tf.distribute.TPUStrategy(cluster_resolver)

            with self.strategy.scope():
                self.model.compile(
                    optimizer=self.config.train.optimizer_1,
                    classification_loss=self.config.train.classification_loss,
                    box_loss=self.config.train.box_loss,
                )
        if self.config.train.accelerator == "cpu":
            self.model.compile(
                optimizer=self.config.train.optimizer_1,
                classification_loss=self.config.train.classification_loss,
                box_loss=self.config.train.box_loss,
            )
