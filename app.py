import argparse
import os
import loguru
import keras_cv
import tensorflow as tf
from tensorflow import keras
from keras_cv import bounding_box
from keras_cv import visualization
import yaml

class StaticDotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = StaticDotDict(value)
            self[key] = value

print = loguru.logger.debug

 
class Dataset:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.arg_parse()

    def arg_parse(self):
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        config = StaticDotDict(config)
        return config

    def configure_dataset(self):
        raise NotImplementedError

    def visualize_dataset(self):
        raise NotImplementedError

class YOLOv8Trainer:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.arg_parse()

    def arg_parse(self):
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        config = StaticDotDict(config)
        return config

    def configure_model(self):
        backbone = keras_cv.models.YOLOV8Backbone.from_preset(self.config.model.backbone)
        yolo_v8_model = keras_cv.models.YOLOV8Detector(
            backbone=backbone,
            num_classes=len(self.config.model.classes),
            bounding_box_format=self.config.model.bounding_box_format,
            fpn_depth=self.config.model.fpn_depth,
        )
        self.model = yolo_v8_model
        
    def configure_optimizers(self):
        raise NotImplementedError

    def configure_callback(self):
        raise NotImplementedError


def main():
    trainer = YOLOv8Trainer('config.yaml')
    model = trainer.configure_model()


if __name__ == '__main__':
    main()
    




