from determined.keras import TFKerasTrial, TFKerasTrialContext, InputData
from model import *


class CVAE(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        self.context = context

    def build_model(self):

        return model

    def build_training_data_loader(self):

        return train_dataset

    def build_validation_data_loader(self):

        return test_dataset
