from transformers import AutoModel

from data_manager.entity_classes.tne_batch import TNEBatch
from model.mlp_module import Multilayer_Classifier
from model.modules.loss_module import get_loss_module
from model.model_parameters import Model_Parameters
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
import torch


class TNEModel(torch.nn.Module):
    def __init__(self, hyper_parameters: Model_Parameters, device_type: Optional[str] = "cuda") -> None:
        """ Init TNE model components and configuration """
        super(TNEModel, self).__init__()
        self.device_type = device_type

        #####################################
        #       Training Parameters         #
        #####################################

        training_parameters = hyper_parameters.training_params
        context_layer_config = hyper_parameters.context_layer
        anchor_mlp_config = hyper_parameters.anchor_mlp
        complement_mlp_config = hyper_parameters.anchor_mlp
        prediction_model_config = hyper_parameters.prediction_model

        #####################################
        #         Model Components          #
        #####################################

        # Context layer which used to gain contextualized representation
        # of each token in the text.
        self.context_layer = AutoModel.from_pretrained(context_layer_config["model"])

        # Anchor mlp used to encode each span as as an anchor,
        # It would represent each span and include the crucial information to help him
        # predict whether the anchor is connected to different component spans
        self.anchor_encoder = Multilayer_Classifier(mlp_config=anchor_mlp_config)

        # Complement mlp used to encode each span as as complement,
        # used to represent each span and include crucial information to help him
        # predict whether the anchor is connected to given complement span
        self.complement_encoder = Multilayer_Classifier(mlp_config=complement_mlp_config)

        # Prediction model is a multiclass classification which receives a
        # concatenated representation for each pair of anchor and complement.
        # and predict the connecting preposition or None is case the NPs are not connected
        self.prediction_model = Multilayer_Classifier(mlp_config=prediction_model_config)

        #####################################
        #           Loss Function           #
        #####################################

        # Loss function which evaluates the model performance.
        self.loss = get_loss_module(training_parameters['loss'])()

    def forward(self, input: TNEBatch, eval_mode: Optional[bool] = False):
        """
        DESCRIPTION: Forward pass
        the calculation process which used to get for each pair of noun
        phrases, probabilities over the prepositions.
        That is, for each ordered pair of non-pronominal base-NP spans in the text (x, y)
        compute the probability that there is a preposition-mediated relation between them.
        and the probability of each preposition to be the connection preposition between them.

        STEPS: The first step is to get the contextualized representation of each span.
        The model take the encoding of each pair of NPs - an anchor and complement.
        concatenate their vectors and feed it into a prediction model.
        The prediction model is a multi-class classifier over the prepositions (or NONE in case the NPs are not connected).
        For each ordered pair of NPs, the model computes the probabilities over aforementioned classes

        ARGUMENTS:
          - input (TNEBatch): batch of samples from the dataset.
          - eval_model (Optional[bool]): training mode or evaluation mode.
        """

        # The input for the context layer is the tokens for the documents in batch.
        # The tokenized_input is composed of two main components:
        # input_ids: The ids of the tokens (dim= batch_size x nof_tokens )
        # attention mask: binary sequence which indicate the position of the padded
        # indices so that the model does not attend to them.
        # dim: [ BATCH_SIZE x NOF_TOKENS ]
        tokenized_input = input.tokens

        # Apply context model on the tokens to encode the tokens in the documents in the
        # batch in order to gain contextualized representation for each token.
        # output dim: [ BATCH_SIZE x SEQ_LEN x FEATURES_DIM ]
        context_output = self.context_layer(input_ids=tokenized_input['input_ids'],
                                            attention_mask=tokenized_input['attention_mask'])
        batch_size, seq_len, features_dim = context_output[0].shape

        # Compute cumulative sum over the contextualized representation of the tokens.
        # output dim = batch_size x seq_len + 1 x features_dim
        # cum_sum_input = torch.cat((context_output[0], torch.zeros(batch_size, 1, features_dim, device="cuda")), dim=1)
        # cum_sum = torch.cumsum(cum_sum_input, dim=1)
        # seq_len += 1

        # Compute sum for every span combination in order to gain
        # contextualized representation for the spans (=noun phrases) by sum up
        # the inner tokens representations.
        # output dim: [ BATCH_SIZE x SEQ_LEN x SEQ_LEN x FEATURES_DIM ]
        tokens_embeddings = context_output[0]
        a = tokens_embeddings.repeat(1, seq_len, 1)
        b = tokens_embeddings.repeat(1, 1, seq_len).view(batch_size, -1, features_dim)
        c = a.sub(b).view(batch_size, seq_len, seq_len, features_dim)

        # For each NP span, accessing the relevant span sum from the representations
        # span_embeddings dim:  [ BATCH_SIZE x NOF_SPANS x FEATURES_DIM ]
        span_embeddings = [pad_sequence([c[batch_idx, span[0], span[1]] for span in batch_spans]) for
                           batch_idx, batch_spans in enumerate(input.spans)]
        span_embeddings = pad_sequence(span_embeddings).view(batch_size, -1, features_dim)

        # Feed each span into the anchor encoder/mlp to gain representations for the spans as an anchors
        # anchor_reps dim: [ BATCH_SIZE x NOF_TOKENS x ANCHOR_REP ]
        anchor_reps = self.anchor_encoder(span_embeddings)

        # Feed each span into the complement encoder/mlp to gain representations for the spans as complements
        # complement_reps dim = batch x nof_spans x complement_rep
        complement_reps = self.complement_encoder(span_embeddings)

        # Creating a large matrix that concatenates all permutations of anchor with complement
        # permutations_mat dim : [ BATCH_SIZE x NOF_SPANS x NOF_SPANS x ANCHOR_REP + COMPLEMENT_REP ]
        # permutations_spans dim : [ 1 x BATCH_SIZE * NOF_SPANS * NOF_SPANS x ANCHOR_REP + COMPLEMENT_REP ]
        batch_size, nof_spans, features_dim = anchor_reps.shape
        permutations_mat = torch.cat([anchor_reps.repeat(1, nof_spans, 1),
                                      complement_reps.repeat(1, 1, nof_spans).view(batch_size, -1, features_dim)], dim=-1)
        permutations_spans = permutations_mat.view(1, -1, 2 * features_dim)

        # Invoke prediction model to get probability for each label/preposition.
        # predicted_labels dim: [ BATCH_SIZE * NOF_SPANS * NOF_SPANS x num_labels ]
        # such that the number of labels is the number of valid prepositions
        predicted_labels = self.prediction_model(permutations_spans).squeeze(0)

        # Compute loss value which gains the probabilities
        loss = self.loss(predicted_labels, input.preposition_labels)

        # Return loss and the logits (probabilities vectors)
        return {"loss": loss, "logits": predicted_labels}
