import argparse
import numpy

from demos.modules import models
from demos.pipelines.base_pipeline import BasePipeline


class Classification(BasePipeline):
    """Placeholder class for classification pipeline
    """

    def __init__(self):
        self.epochs = 10
        variables = numpy.array([])

    def set_model(self) -> None:
        """Set the model architecture
        """

        self.model = models.ModelX()

    def set_args(self, args) -> None:
        """Sets the class variables based on input args
        """

        self.epochs = args.epochs

    def run(self) -> None:
        """Run the pipeline

        Args:
            epochs (int, optional): _description_. Defaults to 10.
        """

        self.set_model()


def main():
    """Main function to parse command line arguements and run pipeline
    """    

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest="epochs", type=int, help='Sets the number of epochs')
    args = parser.parse_args()

    pipeline = Classification()
    pipeline.set_args(args)
    pipeline.run()
    Classification.run()


if __name__ == "__main__":
    main()