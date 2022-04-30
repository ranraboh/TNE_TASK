from model.enums.optimizer import Optimizer
from model.enums.loss_module import Loss_Function
from model.enums.activation_function import Activation_Function

# Model configuration which contains the training hyper parameters for TNE model
class Model_Parameters:
    def __init__(self, num_labels: int) -> None:
        # Number of classes (preposition labels)
        self.num_labels = num_labels

        # Grouping the classes according to their frequency in the training data.
        self.dominant_labels = [0]
        self.low_frequency_labels = [8, 13, 14, 17, 19, 20, 21, 22, 23, 24]
        self.mid_frequency_labels = [2, 4, 5, 6, 7, 9, 10, 15, 16, 18 ]

        # hyper parameters that are related to the training stage
        self.training_params = {
            'epochs': 40,
            'batch_size': 1,
            'learning_rate': 1e-5,
            'weight_decay': 0.01,
            'optimizer': Optimizer.AdamW,
            'loss': Loss_Function.Cross_Entropy,
            # Number of warmup steps for learning rate scheduler
            'warmup_steps': 100,
            # Weights for classes groups
            'dominant_class_weight': 1 / 3,
            'mid_frequency_weight': 2,
            'low_frequency_weight': 4,
            'concrete_labels_rate': 0.01
        }

        # hyper parameters that are related to the evaluation stage
        self.evaluation_params = {
            'batch_size': 1,
            'eval_steps': 1000
        }

        # Configuration for the context layer which used to gain contextualized representation
        # of the tokens in the text
        self.context_layer = {
            'model': 'roberta-large',
            'frozen': False,
            'tokenizer': 'roberta-large',
        }

        # Config for the span extraction encoder
        self.encoder_layer = {
            'input_dim' : 1080,
            'nof_layers': 3,
            'nof_heads': 8,
            'dropout': 0.2,
            'positional_dim': 56,
            'subsequence_levels': [40, 20]
        }

        # Config for anchor mlp which used to encode each span as an anchor
        self.anchor_mlp = {
            'num_layers': 1,
            'layers_dims': [6368, 500],
            'activation_function': Activation_Function.PReLU,
            'dropout_rate': 0.3,
            'batch_norm': False
        }

        # Config for complement mlp which used to encode each span as complement
        self.complement_mlp = {
            'num_layers': 2,
            'layers_dims': [6368, 500],
            'activation_function': Activation_Function.PReLU,
            'dropout_rate': 0.3,
            'batch_norm': False
        }

        # Config for the prediction model which
        self.prediction_model = {
            "num_layers": 2,
            "layers_dims": [1050, 300, num_labels],
            "activation_function": Activation_Function.PReLU,
            "dropout_rate": 0.3,
            "batch_norm": False
        }

        # Config for the coreference resolution graph neural network
        self.graph_convolution = {
            'nof_layers': 2,
            'input_dim': 4208,
            'hidden_dim': 4208,
            'output_dim': 1080,
            'dropout': 0.1
        }

        # Config for the dynamic graph neural network component
        self.dynamic_graph_convolution = {
            'span_compression': {
                'num_layers': 1,
                'layers_dims': [4208, 1080],
                'activation_function': Activation_Function.ReLU,
                'dropout_rate': 0.3,
                'batch_norm': False
            },
            'link_classification': {
                'input_dim': 2160,
                'output_dim': 2,
            },
            'graph_convolution': {
                'nof_layers': 2,
                'input_dim': 1080,
                'hidden_dim': 1080,
                'output_dim': 1080,
                'dropout': 0.1
            }
        }

        # Config of the relation distance embedding
        self.distance_embedding_layer = {
            "embedding_dim": 50,
            "embedding_size": 400 + 1,
            "max_distance": 200
        }
