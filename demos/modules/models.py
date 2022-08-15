from pathlib import Path

import yolov5
from yolov5 import train, val, detect, export


class ModelX():
    """Placeholder class for model x
    """

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

    def train(self):
        """Train the model on a custom dataset
        """

        train.run(imgsz=640, data='coco128.yaml')
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