import argparse
import numpy

from demos.modules import models
from demos.pipelines.base_pipeline import BasePipeline


class Classification(BasePipeline):
    """Class for a folder based classification pipeline
    """

    def __init__(self):
        self.epochs = 10

    def set_model(self) -> None:
        """Set the model architecture
        """

        self.model = models.TensorflowClassification()

    def set_args(self, args) -> None:
        """Sets the class variables based on input args
        """

        self.epochs = args.epochs
        self.output = args.output
        self.input = args.input

    def run(self) -> None:
        """Run the pipeline

        Args:
            outputs (Path): Folder to save the model too
            epochs (int): Number of epochs to train the model
        """

        self.set_model()
        self.model.train_model(self.output, self.epochs)
        self.model.load_model(self.output)
        results = self.model.inference(self.output)


def main():
    """Main function to parse command line arguements and run pipeline
    """    

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest="epochs", type=int, help='Sets the number of epochs')
    parser.add_argument('--output', dest="output", type=str, help='Folder to save the model into')
    parser.add_argument('--input', dest="input", type=str, help='Folder containing the images to run inferences on')
    args = parser.parse_args()

    pipeline = Classification()
    pipeline.set_args(args)
    pipeline.run()


if __name__ == "__main__":
    main()