from data import Data
from model import Model
from pytorch_lightning import Trainer
import os


def main():
    model = Model()
    trainer = Trainer(max_epochs=5)
    trainer.fit(model)

if __name__ == "__main__":
    main()