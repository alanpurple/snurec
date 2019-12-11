""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

File: rec/models.py
- Recommendation model classes and functions for using them.

Version: 1.0.0

refactored by Alan Anderson (alan@wemakeprice.com)

"""
import tensorflow as tf
from tensorflow.keras import layers,activations,Model,regularizers,losses,Sequential
#from tensorflow.keras import backend as K

@tf.function
def get_sequence_length(sequence):
    """
    Calculate the valid length of a sequence (ignoring zeros).

    :param sequence: the sequence.
    :return: the valid length of the sequence.
    """
    abs_seq = tf.abs(sequence)
    used = tf.sign(tf.reduce_max(abs_seq, 2))
    return tf.reduce_sum(used, 1)

class BaselineModel(Model):
    """
    Base model for recommendation, which has no learnable parameters.
    """

    def __init__(self,num_item,item_emb_len, embeddings, mode='average'):
        """
        Class initializer.

        :param embeddings: embedding vectors of all items.
        :param mode: the prediction mode of this model: average or last.
        """
        super().__init__()
        self.item_emb=layers.Embedding(num_item+1,item_emb_len,trainable=False)
        # initialzie embedding matrix before setting
        dummy=self.item_emb(0)
        self.item_emb.set_weights([embeddings])
        self.mode = mode
        self.permute=layers.Permute((2,1))

    @tf.function
    def call(self, inputs):
        """
        Run forward propagation to produce outputs.

        :param inputs: a tuple of input tensors for predictions.
        :param candidates: a tuple of input tensors for candidates.
        :return: the predicted scores for all candidates.
        """
        orders = self.item_emb(inputs)

        if self.mode == 'average':
            out = tf.reduce_sum(orders, axis=1)
            out /= tf.expand_dims(get_sequence_length(orders), axis=1)
        elif self.mode == 'last':
            out = orders[:, -1, :]
        else:
            raise ValueError(self.mode)
        return layers.dot([out, self.permute(self.item_emb.get_weights()[0])])

class RNN1(Model):
    """
    RNN 1 model for recommendation, which uses none of important techniques.
    """

    def __init__(self,num_item,item_emb_len, embeddings, num_units=32, num_layers=1, decay=0):
        """
        Class initializer.

        :param embeddings: embedding vectors of all items.
        :param num_units: the number of hidden units in each LSTM cell.
        :param num_layers: the number of LSTM layers.
        :param decay: an L2 decay parameter for regularization.
        """
        super().__init__()
        self.item_emb=layers.Embedding(num_item+1,item_emb_len,mask_zero=True,trainable=False)
        # initialzie embedding matrix before setting
        dummy=self.item_emb(0)
        self.item_emb.set_weights([embeddings])
        lstm = Sequential()
        for _ in range(num_layers):
            lstm.add(layers.LSTM(num_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(decay),
                                bias_regularizer=regularizers.l2(decay),
                                activation='sigmoid'))
        self.lstm=lstm
        self.linear = layers.Dense(embeddings.shape[1], kernel_regularizer=regularizers.l2(decay),
                                    bias_regularizer=regularizers.l2(decay))
        self.softmax_bias = tf.zeros(embeddings.shape[0])
        self.permute=layers.Permute((2,1))

    @tf.function
    def call(self, inputs):
        """
        Run forward propagation to produce outputs.

        :param inputs: a tuple of input tensors for predictions.
        :param candidates: a tuple of input tensors for candidates.
        :return: the predicted scores for all candidates.
        """
        orders = self.item_emb(inputs)
        orders = self.lstm(orders)
        out = orders[:, -1, :]
        out = self.linear(out)
        return layers.dot([out, self.permute(self.item_emb.get_weights()[0])])

class RNN2(RNN1):
    """
    RNN 2 model for recommendation, which uses category information.
    """
    def __init__(self,num_item,item_emb_len, embeddings, categories, num_units=32,
                 num_layers=1, decay=0, emb_way='mean'):
        """
        Class initializer.

        :param embeddings: embedding vectors of all items.
        :param categories: multi-hot categories of all items.
        :param emb_way: how to get embedding vectors of items.
        :param num_units: the number of hidden units in each LSTM cell.
        :param num_layers: the number of LSTM layers.
        :param decay: an L2 decay parameter for regularization.
        """
        super().__init__(num_item,item_emb_len,embeddings, num_units, num_layers, decay)

        nx = embeddings.shape[1]
        nc = categories.shape[1]

        self.cate_emb=layers.Embedding(categories.shape[0],categories.shape[1],mask_zero=True,trainable=False)
        # initialzie embedding matrix before setting
        dummy=self.cate_emb(0)
        self.cate_emb.set_weights([categories])
        self.cat_embeddings = layers.Embedding(nc, nx,embeddings_regularizer=regularizers.l2(decay),
                                embeddings_initializer='zeros',mask_zero=True)
        self.emb_way = emb_way
        if emb_way == 'mlp':
            self.emb_layer = layers.Dense(nx,'tanh')

        self.softmax=layers.Softmax(1)

    @tf.function
    def call(self, inputs, candidates):
        """
        Run forward propagation to produce outputs.

        :param inputs: a tuple of input tensors for predictions.
        :param candidates: a tuple of input tensors for candidates.
        :return: the predicted scores for all candidates.
        """
        users, orders, clicks = inputs
        orders = self._lookup_features(orders)
        orders = self.lstm(orders)
        clicks = self._lookup_features(clicks)
        out = self._run_attention(users, orders, clicks)
        out = self.linear(out)
        cands_v = self._lookup_candidates()
        return layers.dot([out, self.permute(cands_v)])

    @tf.function
    def _lookup_features(self, seqs):
        """
        Look up the feature vectors of all items in sequences.

        :param seqs: the behavioral sequences for predictions.
        :return: the feature vectors.
        """

        item_embs = self.item_emb(seqs)
        categories = self.cate_emb(seqs)
        categories_sum = tf.reduce_sum(categories, 2, keepdims=True)
        categories = categories / tf.where(categories_sum > 0, categories_sum, 1)
        cat_embeddings = self.cat_embeddings.weights[0]
        cat_embeddings = tf.tensordot(categories, cat_embeddings, axes=[[2], [0]])
        return self._combine_embeddings(item_embs, cat_embeddings)

    @tf.function
    def _lookup_candidates(self):  # N x D
        """
        Look up candidate vectors by embeddings and categories.

        :param embeddings: the embeddings of candidates.
        :param categories: the categories of candidates.
        :return: the chosen vectors.
        """
        categories_sum = tf.reduce_sum(self.cate_emb.get_weights()[0], 1, keepdims=True)
        categories = self.cate_emb.get_weights()[0] / tf.where(categories_sum > 0, categories_sum, 1)
        cat_embeddings = self.cat_embeddings.get_weights()[0]
        cat_embeddings = layers.dot([categories, cat_embeddings])
        return self._combine_embeddings(self.item_emb.get_weights()[0], cat_embeddings)

    @tf.function
    def _combine_embeddings(self, embeddings, cat_embeddings):
        """
        Combine vectors of titles and categories to get representations.

        :param embeddings: embedding vectors learned for titles.
        :param cat_embeddings: embedding vectors for categories.
        :return: the final representations of items.
        """
        if self.emb_way == 'mean':
            return layers.average([embeddings, cat_embeddings])
        elif self.emb_way == 'mlp':
            out = layers.concatenate([embeddings, cat_embeddings], axis=-1)
            return self.emb_layer(out)
        else:
            raise ValueError()

    @tf.function
    def _run_attention(self, users, vec_orders, vec_clicks):
        """
        Run an attention mechanism for getting an output.

        :param users: users in the given inputs.
        :param vec_orders: the output (hidden) vectors of orders.
        :param vec_clicks: the output (hidden) vectors of clicks.
        :return: the chosen vector.
        """
        return vec_orders[:, -1, :]


class RNN3(RNN2):
    """
    RNN 3 model for recommendation, which uses an attention mechanism.
    """

    @tf.function
    def _run_attention(self, users, vec_orders, vec_clicks):
        """
        Run an attention mechanism for getting an output.

        :param users: users in the given inputs.
        :param vec_orders: the output (hidden) vectors of orders.
        :param vec_clicks: the output (hidden) vectors of clicks.
        :return: the chosen vector.
        """
        out_last = vec_orders[:, -1, :]  # the last hidden vector
        tiled = tf.tile(tf.expand_dims(out_last, 1), [1, vec_orders.shape[1], 1])
        scores = tf.reduce_sum(tf.multiply(vec_orders, tiled), axis=2, keepdims=True)
        scores = self.softmax(scores)
        out = tf.reduce_sum(vec_orders * scores, axis=1)  # context vector
        return layers.concatenate([out, out_last], axis=1)


class RNN4(RNN2):
    """
    RNN 3 model for recommendation, which uses clicks for attention keys.
    """

    def __init__(self,num_item,item_emb_len, embeddings, categories, num_layers, num_units, decay,emb_way):
        """
        Class initializer.

        :param embeddings: embedding vectors of all items.
        :param categories: multi-hot categories of all items.
        :param num_layers: the number of LSTM layers.
        :param num_units: the number of hidden units in each LSTM cell.
        :param decay: an L2 decay parameter for regularization.
        """
        super().__init__(num_item,item_emb_len,embeddings, categories, num_layers=num_layers, num_units=num_units)
        self.lstm_click = Sequential()
        for _ in range(num_layers):
            self.lstm_click.add(
                layers.Bidirectional(layers.LSTM(num_units, return_sequences=True,
                                                 kernel_regularizer=regularizers.l2(decay),
                                                 bias_regularizer=regularizers.l2(decay),
                                                 activation='sigmoid')))
        self.linear_click = layers.Dense(num_units,
                                         kernel_regularizer=regularizers.l2(decay),
                                         bias_regularizer=regularizers.l2(decay))

    @tf.function
    def _run_attention(self, x_user, vec_orders, vec_clicks):
        """
        Run an attention mechanism for getting an output.

        :param x_user: users in the given inputs.
        :param vec_orders: the output (hidden) vectors of orders.
        :param vec_clicks: the output (hidden) vectors of clicks.
        :return: the chosen vector.
        """
        out = self.lstm_click(vec_clicks)
        out_last = out[:, -1, :]
        key = self.linear_click(out_last)
        tiled = tf.tile(tf.expand_dims(key, 1), [1, vec_orders.shape[1], 1])
        scores = tf.reduce_sum(tf.multiply(vec_orders, tiled), axis=2, keepdims=True)
        scores = self.softmax(scores)
        out = tf.reduce_sum(vec_orders * scores, axis=1)  # context vector
        return layers.concatenate([out, vec_orders[:, -1, :]], axis=1)
