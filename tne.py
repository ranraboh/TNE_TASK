from data_manager.data_reader import DataReader
from model.model_parameters import Model_Parameters
from trainer.tne_config import TNE_Config
from model.tne_model import TNEModel
from trainer.tne_trainer import TNETrainer
import torch

###############################################
#                Load Dataset                 #
###############################################

torch.cuda.empty_cache()
tne_config = TNE_Config()
data_reader = DataReader(prepositions_list=tne_config.prepositions_list, tokenizer=tne_config.tokenizer)
train_set = data_reader.load_samples(tne_config.train_set)
evaluation_set = data_reader.load_samples(tne_config.evaluation_set)
test_set = data_reader.load_samples(tne_config.test_set, test_mode=True)

###############################################
#           Build Model & Trainer             #
############################################### 

hyper_parameters = Model_Parameters(tne_config.num_labels)
model = TNEModel(config=tne_config, hyper_parameters=hyper_parameters, device_type=tne_config.device)
trainer = TNETrainer(model=model, train_set=train_set, evaluation_set=evaluation_set, test_set=test_set,
                     config=tne_config, hyper_parameters=hyper_parameters)

# Train the model
trainer.train()

# Evaluation step
trainer.evaluate()
