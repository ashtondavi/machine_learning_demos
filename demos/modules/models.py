from winreg import REG_RESOURCE_REQUIREMENTS_LIST
import yolov5

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
        # set model parameters
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # maximum number of detections per image

    def train(self):
        """Train the model on a custom dataset
        """

        #code this function

    def inference(self, image_url:str):
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