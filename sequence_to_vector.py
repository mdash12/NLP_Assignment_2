# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        self._num_layers = num_layers
        self._dropout = dropout
        self._dan_layers = []
        for layer in range(self._num_layers):
            self._dan_layers.append(layers.Dense(self._input_dim, activation='relu'))
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # ...
        batch_size = vector_sequence.shape[0]
        max_tokens = vector_sequence.shape[1]

        if training:
            dropout_mat = tf.random.uniform([batch_size, max_tokens], dtype=tf.float32)
            dropout_mat = tf.cast(tf.greater(dropout_mat, self._dropout), tf.float32)
            sequence_mask = tf.multiply(sequence_mask, dropout_mat)

        ones = tf.ones([1, self._input_dim], tf.float32)
        sequence_mask = ones * tf.reshape(sequence_mask, [batch_size, max_tokens, 1], tf.float32)

        tot_words = tf.reduce_sum(sequence_mask, axis=1)
        vector_sequence = tf.divide(tf.reduce_sum(tf.multiply(vector_sequence, sequence_mask), axis=1), tot_words)

        x = []
        for layer_num in range(self._num_layers):
            vector_sequence = self._dan_layers[layer_num](vector_sequence)
            x.append(vector_sequence)

        layer_representations = tf.stack(x, axis=1)
        combined_vector = layer_representations[:, -1, :]

        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        # ...
        self.num_layers = num_layers
        self.gru_layers = []
        for layer in range(self.num_layers):
            self.gru_layers.append(layers.GRU(self._input_dim, return_sequences=True, return_state=True))
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        # ...

        representations = []
        states = []
        representation, state = self.gru_layers[0](vector_sequence, mask=sequence_mask)
        representations.append(representation)
        states.append(state)
        for layer_num in range(1, self.num_layers):
            representation, state = self.gru_layers[layer_num](representations[-1], mask=sequence_mask)
            representations.append(representation)
            states.append(state)

        layer_representations = tf.stack(states, axis=1)
        combined_vector = layer_representations[:, -1, :]
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
