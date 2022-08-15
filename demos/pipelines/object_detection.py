import argparse
import os
import pathlib

from demos.modules import models
from demos.pipelines.base_pipeline import BasePipeline


class ObjectDetection(BasePipeline):
    """Placeholder class for a folder based ObjectDetection pipeline
    """

    def __init__(self):
        self.epochs = 100

    def set_model(self) -> None:
        """Set the model architecture
        """

        self.model = models.Yolo5()
        self.model.load_default_model()

    def set_args(self, args) -> None:
        """Sets the class variables based on input args
        """

        self.epochs = args.epochs

    def run(self, path_images:str) -> None:
        """Run the pipeline

        Args:
            path_images (str): Folder of images to process
        """


        self.set_model()
        #self.model.train()
        for file in os.listdir(path_images):
            print(file)
            path_image = pathlib.Path(path_images, file)
            predictions = self.model.inference(path_image)
            print(predictions)


def main():
    """Main function to parse command line arguements and run pipeline
    """    

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest="epochs", type=int, help='Sets the number of epochs')
    parser.add_argument('--folder', dest="folder", type=str, help='Folder of images to process')
    args = parser.parse_args()

    pipeline = ObjectDetection()
    pipeline.set_args(args)
    pipeline.run(args.folder)


if __name__ == "__main__":
    main()
