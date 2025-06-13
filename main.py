from callbacks import *
from data import GGWDData
from transformer import TransformerModel
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger 


def main():
    cli = LightningCLI(TransformerModel, GGWDData)


if __name__ == "__main__":
    main()