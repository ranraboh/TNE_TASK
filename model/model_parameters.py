from model.modules.optimizer import Optimizer
from model.modules.loss_module import Loss_Function
from model.modules.activation_function import Activation_Function


# Model configuration which contains the training hyper parameters for TNE model
class Model_Parameters:
    def __init__(self, num_labels: int) -> None:
        self.num_labels = num_labels

        self.training_params = {
            'epochs': 10,
            'batch_size': 1,
            'learning_rate': 1e-5,
            'weight_decay': 0.01,
            'optimizer': Optimizer.AdamW,
            "loss": Loss_Function.Cross_Entropy,
            # Number of warmup steps for learning rate scheduler
            'warmup_steps': 100
        }

        self.evaluation_params = {
            'batch_size': 1,
            'eval_steps': 500
        }

        # Configuration for the context layer which used to gain contextualized representation
        # of the tokens in the text
        self.context_layer = {
            'model': 'SpanBERT/spanbert-base-cased',
            'frozen': False,
            'tokenizer': 'SpanBERT/spanbert-base-cased'
        }

        self.encoder_layer = {
            'num_encoder_layers': 12,
            'heads': 8,
            'dropout': 0.1,
        }

        # Config for anchor mlp which used to encode each span as an anchor
        self.anchor_mlp = {
            'num_layers': 1,
            'layers_dims': [1536, 500],
            'activation_function': Activation_Function.PReLU,
            'dropout_rate': 0.1,
            'batch_norm': False
        }

        # Config for complement mlp which used to encode each span as complement
        self.complement_mlp = {
            'num_layers': 1,
            'layers_dims': [1536, 500],
            'activation_function': Activation_Function.PReLU,
            'dropout_rate': 0.1,
            'batch_norm': False
        }

        # Config for the prediction model which
        self.prediction_model = {
            "num_layers": 2,
            "layers_dims": [1000, 500, num_labels],
            "activation_function": Activation_Function.PReLU,
            "dropout_rate": 0.1,
            "batch_norm": False
        }