from model.tne_model import TNEModel
import torch

class Model_Saver:
    def __init__(self, model : TNEModel, path : str):
        self.model = model
        self.model_path = path

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_best_model(self):
        parameters = torch.load(self.model_path)
        self.model.load_state_dict(parameters)