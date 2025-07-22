from callbacks.default import *
from callbacks.mha import *
from data import GGWDData
from models.mha import MHAModel
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.cli import LightningCLI


def main():
    cli = LightningCLI(MHAModel, GGWDData)


if __name__ == "__main__":
    main()