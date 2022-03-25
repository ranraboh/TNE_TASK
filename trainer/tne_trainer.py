from torch.optim import optimizer
from torch.utils.data import Dataset
from transformers import Trainer
from transformers import TrainingArguments
from transformers import get_linear_schedule_with_warmup


from model.modules.optimizer import get_optimizer_module
from trainer.callbacks.printer import PrinterCallback
from data_manager.batch_sampler import Batch_Sampler
from model.model_parameters import Model_Parameters
from model.tne_config import TNE_Config
from trainer.metrics import compute_metrics
import torch.optim

import torch
import os
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
                                               #load_best_model_at_end=True,
                                               # load the best model when finished training (default metric is loss)
                                               logging_dir=config.logs_dir,
                                               logging_steps=5000,  # log & save weights each logging_steps
                                               evaluation_strategy="steps", # evaluate each `logging_steps`
                                               eval_steps=evaluation_params['eval_steps'],
                                               save_strategy="no")

        #############################################
        #               Init Trainer                #
        #############################################

        #optimizer = torch.optim.AdamW(self.model.parameters(), lr=training_params['learning_rate'], weight_decay=training_params['weight_decay'])
        #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_params["warmup_steps"], last_epoch=-1)

        # Metrics
        self.batch_collator = Batch_Sampler(tokenizer_type=hyper_parameters.context_layer['tokenizer'],
                                       device_type=self.config.device)
        self.trainer = Trainer(
            model=self.model,                                   # TNE model
            args=self.training_args,                            # Training arguments, defined above
            train_dataset=self.train_set,                       # Training dataset
            eval_dataset=self.evaluation_set,                   # Evaluation dataset
            #compute_metrics=self.metrics.compute_metrics,       # Callback that computes metrics of interest
            callbacks=[
                # a printer callback used to draw a graph showing the
                # evaluation accuracy of the model over the epochs in the training.
                PrinterCallback
            ],
            #optimizers=(optimizer, scheduler),
            data_collator=self.batch_collator,
            compute_metrics=compute_metrics,
        )

    def train(self):
        # train the model
        self.trainer.train()

    def evaluate(self):
        # evaluate the current model after training
        self.trainer.evaluate()

    """
    def compute_metrics(pred):
      labels = pred.label_ids
      preds = pred.predictions.argmax(-1)
      # calculate accuracy using sklearn's function
      acc = accuracy_score(labels, preds)
      return {
        'accuracy': acc,
      }
    """
    