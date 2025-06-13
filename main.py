from callbacks import *
from data import GGWDData
from model import Conv1dModel
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger 


def main():
    cli = LightningCLI(Conv1dModel, GGWDData)


if __name__ == "__main__":
    main()