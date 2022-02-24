from data_manager.data_reader import DataReader
from model.model_parameters import Model_Parameters
from model.tne_config import TNE_Config
from model.tne_model import TNEModel
from trainer.tne_trainer import TNETrainer

###############################################
#                Load Dataset                 #
###############################################

tne_config = TNE_Config()
data_reader = DataReader(prepositions_list=tne_config.prepositions_list, device_type=tne_config.device)
evaluation_set = data_reader.load_samples(tne_config.evaluation_set)
# create train set as well.

###############################################
#           Build Model & Trainer             #
############################################### 

hyper_parameters = Model_Parameters(tne_config.num_labels)
model = TNEModel(hyper_parameters=hyper_parameters, device_type=tne_config.device)
trainer = TNETrainer(model=model, train_set=evaluation_set, evaluation_set=evaluation_set, test_set=evaluation_set, 
                     config=tne_config, hyper_parameters=hyper_parameters)

# Train the model
trainer.train()

# Evaluation step
trainer.evaluate()

# Save params/weights/model/components

