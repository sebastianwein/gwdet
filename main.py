from data import Data
import os
from model import Conv1dModel
import numpy as np
from pytorch_lightning import Trainer
from torchinfo import summary


def main():
    batch_size = 64
    data = Data("/scratch/tmp/swein/ggwd/output/bbh.hdf", batch_size=batch_size, num_workers=4)
    input_size = np.prod(data.sample_shape)
    model = Conv1dModel(learning_rate=0.0005)
    summary(model, input_size=(batch_size, *data.sample_shape))
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, data)


if __name__ == "__main__":
    main()