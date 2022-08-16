#disable tensorflow logging when not using cuda integration
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import json
import yolov5
from yolov5 import train, val, detect, export
import tensorflow as tf


class TensorflowClassification():
    """A tensorflow classification model
    """

    def __init__(self):
        self.model = None
        self.class_names = None

    def load_dataset(self, path_data:Path) -> None:
        """Load a custom dataset for training a model

        Args:
            path_data (Path): _description_
        """

        #code this function...

    def train_model(self, path_model:Path, epochs:int) -> None:
        """Trains a classification model using tensorflow

        Args:
            path_model (Path): Path to save the trained model too
            epochs (int): Number of epochs to train the model for
        """

        #load default dataset >>> replace this with a dataset load function and preprocessing
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        print(train_images.shape)

        #preprocess images
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        #define model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)])
        #add optimizers and loss functions
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        #train model
        model.fit(train_images, train_labels, epochs=epochs)
        #evaluate on test dataset
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)
        #save the model
        model.save(path_model)
        #save class names
        with open(os.path.join(path_model, "class_names.json"), "w") as file:
            json.dump(self.class_names, file)

    def load_model(self, path_model:Path) -> None:
        """Load a saved classification model

        Args:
            path_model (Path): The directory of the selected model
        """

        model = tf.keras.models.load_model(path_model)
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        with open(os.path.join(path_model, "class_names.json"), "r") as file:
            self.class_names = json.load(file)
        self.model = probability_model

    def inference(self, image_url:Path) -> list:
        """Generate an inference for the specified input image

        Args:
            image_url (Path): The path or url to the image for inference

        Returns:
            predictions (list): The inference results for the image
        """        

        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        predictions = self.model.predict(test_images)
        print(self.class_names)
        print(predictions[0])
        return predictions[0]

class Yolo5():
    """Placeholder for class to link to Yolo5

    NOTE: This class is mostly unnecessary due to the tidy packaging of yolov5.
    Just setting it up as an exploration of class structure.
    """

    def __init__(self):
        self.model = None

    def load_default_model(self):
        """Load the default yolov5 model weights
        """

        self.model = yolov5.load('yolov5s.pt')
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image

    def train_model(self, batch_size:int=2, epochs:int=100) -> None:
        """Train the model on a custom dataset

        Args:
            batch_size (int, optional): _description_. Defaults to 2.
            epochs (int, optional): _description_. Defaults to 100.
        """

        train.run(imgsz=640, data='coco128.yaml', batch_size=batch_size, epochs=epochs)
        val.run(imgsz=640, data='coco128.yaml', weights='yolov5s.pt')
        detect.run(imgsz=640)
        export.run(imgsz=640, weights='yolov5s.pt')

    def inference(self, image_url:Path) -> list:
        """Get inferences for a selected image

        Args:
            image_url (str): _description_

        Returns:
            results (???): The inferences from the image
        """        

        results = self.model(image_url)
        results.show()
        predictions = results.pred[0]
        boxes = predictions[:, :4]
        return boxes