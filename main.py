from callbacks import ConfusionMatrix, ROC, SNRAccuracy
from data import Data
from model import Conv1dModel
from pytorch_lightning.cli import LightningCLI

def main():
    cli = LightningCLI(Conv1dModel, Data)

if __name__ == "__main__":
    main()