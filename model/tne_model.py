from transformers import RobertaConfig, RobertaForMaskedLM
from data_manager.entity_classes.tne_batch import TNEBatch
from model.graph_modules.dynamic_graph import DynamicGraphModule
from model.graph_modules.graph_convolution import GraphConvolution
from model.hierarchical_encoder import HierarchicalEncoder
from model.modules.mlp_module import Multilayer_Classifier
from model.enums.loss_module import get_loss_module
from model.model_parameters import Model_Parameters
from typing import Optional
import torch
from trainer.metrics import MetricsEvaluation
import random

class TNEModel(torch.nn.Module):
    def __init__(self, config, hyper_parameters: Model_Parameters, device_type: Optional[str] = "cuda") -> None:
        """ Init TNE model components and configuration """
        super(TNEModel, self).__init__()
        self.device_type = device_type
        self.optimization_steps = 0
        self.tne_config = config

        #####################################
        #       Training Parameters         #
        #####################################

        training_parameters = hyper_parameters.training_params
        context_layer_config = hyper_parameters.context_layer
        anchor_mlp_config = hyper_parameters.anchor_mlp
        complement_mlp_config = hyper_parameters.anchor_mlp
        prediction_model_config = hyper_parameters.prediction_model
        encoder_layer_config = hyper_parameters.encoder_layer
        graph_convolution_config = hyper_parameters.graph_convolution
        dynamic_graph_net_config = hyper_parameters.dynamic_graph_convolution
        distance_embeddings_config = hyper_parameters.distance_embedding_layer
        self.concrete_labels_rate = training_parameters['concrete_labels_rate']
        self.max_spans_distance = distance_embeddings_config['max_distance']

        #####################################
        #         Model Components          #
        #####################################

        # Context layer which used to gain contextualized representation
        # of each token in the text.
        context_config = RobertaConfig.from_pretrained(context_layer_config["model"], output_hidden_states=True)
        self.context_layer = RobertaForMaskedLM.from_pretrained(context_layer_config["model"], config=context_config)
        self.context_layer.resize_token_embeddings(context_config.vocab_size + 2)

        # Embedding for the relative distance between NP pairs.
        # The embedded distance is inject into the model before the output layer which produces the probabilities.
        self.distance_embeddings = torch.nn.Embedding(num_embeddings=distance_embeddings_config['embedding_size'], embedding_dim=distance_embeddings_config['embedding_dim'])

        # The module hierarchically refines the encoding of the tokens to gain a better representation.
        # and extract representation for each span in the document.
        self.hierarchical_encoder = HierarchicalEncoder(config=encoder_layer_config)

        # graph neural network which used to leverage the information regarding the co-reference cluster
        # and encode them into the model.
        # The graph neural network produces a representation for the spans that is mindful of the co-reference clusters
        self.graph_convolution = GraphConvolution(config=graph_convolution_config)

        # The dynamic graph neural network components is used share infromation between the spans.
        # The graph neural network learns the connections which map out the way the information is conveyed between the spans.
        self.dynamic_graph_convolution = DynamicGraphModule(config=dynamic_graph_net_config)

        # Anchor mlp used to encode each span as as an anchor,
        # It would represent each span and include the crucial information to help him
        # predict whether the anchor is connected to different component spans
        self.anchor_encoder = Multilayer_Classifier(mlp_config=anchor_mlp_config, output_probs=False)

        # Complement mlp used to encode each span as as complement,
        # used to represent each span and include crucial information to help him
        # predict whether the anchor is connected to given complement span
        self.complement_encoder = Multilayer_Classifier(mlp_config=complement_mlp_config, output_probs=False)

        # Prediction model is ad multiclass classification which receives a
        # concatenated representation for each pair of anchor and complement.
        # and predict the connecting preposition or None is case the NPs are not connected
        self.prediction_model = Multilayer_Classifier(mlp_config=prediction_model_config, output_probs=True)

        #####################################
        #           Loss Function           #
        #####################################

        # Assign weights for the classes for the loss function.
        # Used to tackle the problem for imbalanced dataset.
        class_weights = torch.ones(hyper_parameters.num_labels)
        class_weights[hyper_parameters.dominant_labels] *= training_parameters['dominant_class_weight']
        class_weights[hyper_parameters.low_frequency_labels] *= training_parameters['low_frequency_weight']
        class_weights[hyper_parameters.mid_frequency_labels] *= training_parameters['mid_frequency_weight']

        # Loss function which evaluates the model performance.
        self.loss = get_loss_module(training_parameters['loss'])(weight=class_weights)
        self.metrics_evaluation  = MetricsEvaluation(self.tne_config)

    def forward(self, input: TNEBatch, labels):
        """
        DESCRIPTION: Forward pass
        the calculation process which used to get for each pair of noun phrases, probabilities over the prepositions list.
        STEPS: The first step is to get the contextualized representation of each span using pre-trained model.
        After obtaining contextual embeddings of the tokens, the data is passed into the span extractor encoder.
        which output a representation for the spans.
        The graph-based methods is used to extract further information which adds up to the initial encoding of the spans
        to get a richer representation. The model take the encoding of each pair of NPs - an anchor and complement.
        concatenate their vectors along with their relative distance and feed it into a prediction model.
        The prediction model is a multi-class classifier over the prepositions (or NONE in case the NPs are not connected).
        For each ordered pair of NPs, the model computes the probabilities over aforementioned classes

        ARGUMENTS:
          - input (TNEBatch): batch of samples from the dataset.
          - eval_model (Optional[bool]): training mode or evaluation mode.
        """
        input.to(self.device_type)
        if self.training:
            self.optimization_steps = self.optimization_steps + 1

        # The input for the context layer is the tokens for the documents in batch.
        # The tokenized_input is composed of two main components:
        # input_ids: The ids of the tokens
        # attention mask: binary sequence which indicate the position of the padded
        # indices so that the model does not attend to them.
        # dim: [ BATCH_SIZE x NOF_TOKENS ]
        tokenized_input = input.tokens

        # Apply context model on the tokens to encode the tokens in the documents in the
        # batch in order to gain contextualized representation for each token.
        # output dim: [ BATCH_SIZE x SEQ_LEN x FEATURES_DIM ]
        context_output = self.context_layer(input_ids=tokenized_input['input_ids'],
                                            attention_mask=tokenized_input['attention_mask'])

        # Compute sum for every span combination in order to gain
        # contextualized representation for the spans (=noun phrases) by sum up
        # the inner tokens representations.
        # output dim: [ BATCH_SIZE x SEQ_LEN x FEATURES_DIM ]
        tokens_embeddings = context_output[1][-1] # last hidden state

        # Span extractor encoder that hierarchically refines the encoding of the tokens to gain a better representation
        # and extract representation for the spans.
        # span_embeddings dim:  [ BATCH_SIZE x NOF_SPANS x INITIAL_SPAN_DIM ]
        span_embeddings = self.hierarchical_encoder(tokens_embeddings, input.spans)

        # leverage the external data regarding co-reference clusters to add up essential information that leads to
        # a more informative, clear, and self-contained representation.
        # Use graph neural network to produce a representation that is mindful of the co-reference clusters and
        # beneficial for evaluating the correct preposition relations between NPs in the text.
        x4 = self.graph_convolution(span_embeddings.squeeze(0), input.coreference_links.squeeze(0)).unsqueeze(0)

        # Dynamic graph neural network is employed to learn the connections which map out the way the information is conveyed between the spans.
        # The goal is to enrich the initial representation with valuable information that is shared among the spans to improve the predictive ability of the model.
        x5 = self.dynamic_graph_convolution(span_embeddings)

        # concatenate the information gain by the graph-based methods to the initial encoding to get a richer representation.
        # span_embeddings dim:  [ BATCH_SIZE x NOF_SPANS x ENHANCED_SPAN_DIM ]
        enhanced_span_embeddings = torch.concat((span_embeddings, x4, x5), dim=-1)

        # Feed each span into the anchor encoder/mlp to gain representations for the spans as an anchors
        # anchor_reps dim: [ BATCH_SIZE x NOF_SPANS x ANCHOR_REP ]
        anchor_reps = self.anchor_encoder(enhanced_span_embeddings)

        # Feed each span into the complement encoder/mlp to gain representations for the spans as complements
        # complement_reps dim = [ BATCH_SIZE x NOF_SPANS x COMPLEMENT_REP ]
        complement_reps = self.complement_encoder(enhanced_span_embeddings)

        # Creating a large matrix that concatenates all permutations of anchor with complement along with their relative distance
        # permutations_mat dim : [ BATCH_SIZE x NOF_SPANS x NOF_SPANS x ANCHOR_REP + COMPLEMENT_REP + RELATIVE_EMBEDDING_DIM ]
        # permutations_spans dim : [ NOF_SPANS * NOF_SPANS x ANCHOR_REP + COMPLEMENT_REP + RELATIVE_EMBEDDING_DIM ]
        batch_size, nof_spans, features_dim = anchor_reps.shape
        starts = input.spans[:, :, 1].squeeze(0)
        spans_distance = starts.repeat(nof_spans) - starts.view(nof_spans, 1).repeat(1, nof_spans).view(-1)
        spans_distance = torch.clamp(spans_distance, -1 * self.max_spans_distance, self.max_spans_distance) + self.max_spans_distance  # hyper max_spans_distance...
        permutations_mat = torch.cat([anchor_reps.repeat(1, nof_spans, 1),
                                      complement_reps.repeat(1, 1, nof_spans).view(batch_size, -1, features_dim),
                                      self.distance_embeddings(spans_distance.unsqueeze(0))], dim=-1)
        permutations_spans = permutations_mat.view(nof_spans * nof_spans, -1)

        # Invoke prediction model
        # to get probability for each label/preposition.
        # predicted_labels dim: [  NOF_SPANS * NOF_SPANS x NUM_LABELS ]
        # such that the number of labels is the number of valid prepositions
        logits = self.prediction_model(permutations_spans).squeeze(0)

        # Print to log the predicted labels, used for debugging
        if self.training and self.optimization_steps % 400 == 0:
            print("--predicted--")
            print (logits.argmax(dim=-1))
            print ("--labels--")
            print (labels)

        # Test mode, returns the predicted prepositions for each NP pair.
        if input.test_mode:
            return logits.argmax(dim=-1)

        # Eval mode, evaluate the metrics and print out the results.
        if not self.training:
            self.metrics_evaluation.evaluate_metrics(input.preposition_labels, logits)

        # Compute loss value which gains the probabilities
        if not self.training or random.random() > self.concrete_labels_rate:
            loss = self.loss(logits, input.preposition_labels)
        else:
            logits = logits[input.concrete_idx][0]
            loss = self.loss(logits, input.concrete_labels.squeeze(0))
        input.to("cpu")

        # Return loss and the logits (probabilities vectors)
        return {"loss": loss, "logits": logits}