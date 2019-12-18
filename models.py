""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

File: rec/models.py
- Recommendation model classes and functions for using them.

Version: 1.0.0

rewritten by Alan Anderson (alan@wemakeprice.com)

"""
import tensorflow as tf
from tensorflow import keras

class RNN1(keras.Model):
    """
    RNN 1 model for recommendation, which uses none of important techniques.
    """

    def __init__(self,emb_size, num_units=32, num_layers=1, decay=0):
        """
        Class initializer.

        :param embeddings: embedding vectors of all items.
        :param num_units: the number of hidden units in each LSTM cell.
        :param num_layers: the number of LSTM layers.
        :param decay: an L2 decay parameter for regularization.
        """
        super().__init__()
        if num_layers<2:
            self.lstm=keras.layers.LSTM(num_units, return_sequences=True,
                                kernel_regularizer=keras.regularizers.l2(decay),
                                bias_regularizer=keras.regularizers.l2(decay),
                                activation='sigmoid',name='lstm')
        else:
            self.lstm=[keras.layers.LSTM(num_units, return_sequences=True,
                                kernel_regularizer=keras.regularizers.l2(decay),
                                bias_regularizer=keras.regularizers.l2(decay),
                                activation='sigmoid',name='lstm{}'.format(i))
                                for i in range(num_layers)]

        self.dense_final = keras.layers.Dense(emb_size, kernel_regularizer=keras.regularizers.l2(decay),
                                    bias_regularizer=keras.regularizers.l2(decay),name='dense_final')

    @tf.function
    def call(self, inputs):
        """
        Run forward propagation to produce outputs.

        :param inputs: a tuple of input tensors for predictions.
        :param candidates: a tuple of input tensors for candidates.
        :return: the predicted scores for all candidates.
        """
        # B X T X E
        orders = inputs['items']
        mask=inputs['mask']
        # B X T X U
        if not isinstance(self.lstm, list):
            orders = self.lstm(orders,mask=mask)
        else:
            for lstm in self.lstm:
                orders=lstm(orders,mask=mask)
        # B X U
        out = orders[:, -1, :]
        # out: B x E
        return self.linear(out)

class RNN2(RNN1):
    """
    RNN 2 model for recommendation, which uses category information.
    """
    def __init__(self, emb_size, num_units=32,
                 num_layers=1, decay=0):
        """
        Class initializer.

        :param num_units: the number of hidden units in each LSTM cell.
        :param num_layers: the number of LSTM layers.
        :param decay: an L2 decay parameter for regularization.
        """
        super().__init__(emb_size,num_units, num_layers, decay)

        self.order_dense = keras.layers.Dense(emb_size,name='order_dense')
        self.click_dens=keras.layers.Dense(emb_size,name='click_dense')
        self.softmax=keras.layers.Softmax(1)

    @tf.function
    def call(self, inputs):
        """
        Run forward propagation to produce outputs.

        :param inputs: a tuple of input tensors for predictions.
        :param candidates: a tuple of input tensors for candidates.
        :return: the predicted scores for all candidates.
        """
        orders=keras.layers.concatenate([inputs['items'],inputs['cate']])
        orders=self.order_dense(orders)
        orders = self.lstm(orders,mask=inputs['mask'])
        clicks=keras.layers.concatenate([inputs['clicks'],inputs['clicks_cate']])
        clicks = self.click_dense(clicks)
        out = self._run_attention(orders, clicks,inputs['clicks_mask'])
        return self.dense_final(out)
        #cands_v = self._lookup_candidates()

    @tf.function
    def _lookup_features(self, item_embs,cat_embs):
        """
        Look up the feature vectors of all items in sequences.

        :param seqs: the behavioral sequences for predictions.
        :return: the feature vectors.
        """
        categories_sum = tf.reduce_sum(cat_embs, 2, keepdims=True)
        categories = cat_embs / tf.where(categories_sum > 0, categories_sum, 1)
        cat_embeddings = self.cat_dense(categories)
        return keras.layers.average([item_embs, cat_embeddings])

    # @tf.function
    # def _lookup_candidates(self,item_embs,cate_embs):
    #     """
    #     Look up candidate vectors by embeddings and categories.

    #     :param embeddings: the embeddings of candidates.
    #     :param categories: the categories of candidates.
    #     :return: the chosen vectors.
    #     """
    #     categories_sum = tf.reduce_sum(cate_embs, 1, keepdims=True)
    #     categories = cate_embs / tf.where(categories_sum > 0, categories_sum, 1)
    #     cat_dense=self.cat_dense(categories)
    #     return keras.layers.average([self.item_emb.weights[0], cat_dense])

    @tf.function
    def _run_attention(self,orders,clicks,clicks_mask):
        """
        Run an attention mechanism for getting an output.

        :param vec_orders: the output (hidden) vectors of orders.
        :param vec_clicks: the output (hidden) vectors of clicks.
        :return: the chosen vector.
        """
        return orders[:, -1, :]


class RNN3(RNN2):
    """
    RNN 3 model for recommendation, which uses an attention mechanism.
    """

    @tf.function
    def _run_attention(self, orders, clicks,clicks_mask):
        """
        Run an attention mechanism for getting an output.

        :param vec_orders: the output (hidden) vectors of orders.
        :param vec_clicks: the output (hidden) vectors of clicks.
        :return: the chosen vector.
        """
        out_last = orders[:, -1, :]  # the last hidden vector
        tiled = tf.tile(tf.expand_dims(out_last, 1), [1, orders.shape[1], 1])
        scores = tf.reduce_sum(tf.multiply(orders, tiled), axis=2, keepdims=True)
        scores = self.softmax(scores)
        out = tf.reduce_sum(orders * scores, axis=1)  # context vector
        return keras.layers.concatenate([out, out_last], axis=1)


class RNN4(RNN2):
    """
    RNN 3 model for recommendation, which uses clicks for attention keys.
    """

    def __init__(self, emb_size, num_layers, num_units, decay):
        """
        Class initializer.

        :param embeddings: embedding vectors of all items.
        :param categories: multi-hot categories of all items.
        :param num_layers: the number of LSTM layers.
        :param num_units: the number of hidden units in each LSTM cell.
        :param decay: an L2 decay parameter for regularization.
        """
        super().__init__(emb_size, num_layers=num_layers, num_units=num_units)
        if num_layers<2:
            self.lstm_click=keras.layers.Bidirectional(keras.layers.LSTM(num_units, return_sequences=True,
                                                 kernel_regularizer=keras.regularizers.l2(decay),
                                                 bias_regularizer=keras.regularizers.l2(decay),
                                                 activation='sigmoid',name='lstm_click'))
        else:
            self.lstm_click=[
                keras.layers.Bidirectional(keras.layers.LSTM(num_units, return_sequences=True,
                                                 kernel_regularizer=keras.regularizers.l2(decay),
                                                 bias_regularizer=keras.regularizers.l2(decay),
                                                 activation='sigmoid',name='lstm_click{}'.format(i)))
                            for i in range(num_layers)]
        self.linear_click = keras.layers.Dense(num_units,
                                         kernel_regularizer=keras.regularizers.l2(decay),
                                         bias_regularizer=keras.regularizers.l2(decay),name='linear_click')

    @tf.function
    def _run_attention(self, orders, clicks,click_mask):
        """
        Run an attention mechanism for getting an output.

        :param vec_orders: the output (hidden) vectors of orders.
        :param vec_clicks: the output (hidden) vectors of clicks.
        :return: the chosen vector.
        """
        if not isinstance(self.lstm_click,list):
            out = self.lstm_click(clicks,mask=click_mask)
        else:
            out=clicks
            for layer in self.lstm_click:
                out=layer(out,mask=click_mask)
        out_last = out[:, -1, :]
        key = self.linear_click(out_last)
        tiled = tf.tile(tf.expand_dims(key, 1), [1, orders.shape[1], 1])
        scores = tf.reduce_sum(tf.multiply(orders, tiled), axis=2, keepdims=True)
        scores = self.softmax(scores)
        out = tf.reduce_sum(orders * scores, axis=1)  # context vector
        return keras.layers.concatenate([out, orders[:, -1, :]], axis=1)
