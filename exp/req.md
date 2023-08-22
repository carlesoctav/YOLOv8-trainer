1. yolov8 model selector
2. hyperparameter setting 
SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0
custom metiic
3. data loader
bounding_boxes = {
    # num_boxes may be a Ragged dimension
    'boxes': Tensor(shape=[batch, num_boxes, 4]),
    'classes': Tensor(shape=[batch, num_boxes])
}

4. {"images": images, "bounding_boxes": bounding_boxes}
5. Augmenter
6. 



