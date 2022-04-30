from torch.utils.data import Dataset
from transformers import Trainer
from transformers import TrainingArguments
from trainer.callbacks.printer import PrinterCallback
from data_manager.batch_sampler import Batch_Sampler
from model.model_parameters import Model_Parameters
from trainer.tne_config import TNE_Config
import torch
import os
import json

os.environ["WANDB_DISABLED"] = "true"


class TNETrainer():
    def __init__(self, model: torch.nn.Module, train_set: Dataset, evaluation_set: Dataset, test_set: Dataset,
                 config: TNE_Config, hyper_parameters: Model_Parameters) -> None:
        # Init Trainer properties
        self.model = model
        self.config = config
        self.prepositions_list = config.prepositions_list
        self.num_labels = config.num_labels

        #################################################
        #                Init TNE Model                 #
        #################################################

        self.train_set = train_set
        self.evaluation_set = evaluation_set
        self.test_set = test_set
        self.test_output_path = self.config.test_output
        self.hyper_parameters = hyper_parameters
        self.model = model

        #################################################
        #           Init Training Arguments             #
        #################################################

        training_params = hyper_parameters.training_params
        evaluation_params = hyper_parameters.evaluation_params
        self.training_args = TrainingArguments(output_dir=config.output_dir,
                                               num_train_epochs=training_params["epochs"],
                                               per_device_train_batch_size=training_params['batch_size'],
                                               per_device_eval_batch_size=evaluation_params['batch_size'],
                                               learning_rate=training_params['learning_rate'],
                                               weight_decay=training_params['weight_decay'],
                                               warmup_steps=training_params['warmup_steps'],
                                               logging_dir=config.logs_dir,
                                               logging_steps=5000,  # log & save weights each logging_steps
                                               evaluation_strategy="steps", # evaluate each `logging_steps`
                                               eval_steps=evaluation_params['eval_steps'],
                                               save_strategy="no")

        #############################################
        #               Init Trainer                #
        #############################################

        # Metrics
        self.batch_collator = Batch_Sampler(tokenizer=self.config.tokenizer,
                                       device_type=self.config.device)
        self.trainer = Trainer(
            model=self.model,                                   # TNE model
            args=self.training_args,                            # Training arguments, defined above
            train_dataset=self.train_set,                       # Training set
            eval_dataset=self.evaluation_set,                   # Evaluation set
            #compute_metrics=self.metrics.compute_metrics,      # Callback that computes metrics of interest
            callbacks=[
                # a printer callback used to draw a graph showing the
                # evaluation accuracy of the model over the epochs in the training.
                PrinterCallback
            ],
            data_collator=self.batch_collator,
        )

    def train(self):
        # train the model
        self.trainer.train()

    def evaluate(self):
        # evaluate the model performance
        self.trainer.evaluate()

    def test(self):
        # test the model and create a file with the predicted prepositions.
        with open(self.test_output_path, 'w') as outfile:
            for sample in self.test_set:
                batch = self.batch_collator.__call__(batch=[sample])
                predictions = self.model(batch['input'], None)
                predictions[predictions == 25] = 0
                predictions_json = json.dumps({'predicted_prepositions': predictions.flatten().tolist()})
                outfile.write(predictions_json + "\n")
