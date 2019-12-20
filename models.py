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
from tensorflow_core.python.keras import Model,layers,regularizers,Input,initializers

class RNN1(Model):
    """
    RNN 1 model for recommendation, which uses none of important techniques.
    """

    def __init__(self,embeddings, num_units=32, num_layers=1, decay=0):
        """
        Class initializer.

        :param embeddings: embedding vectors of all items.
        :param num_units: the number of hidden units in each LSTM cell.
        :param num_layers: the number of LSTM layers.
        :param decay: an L2 decay parameter for regularization.
        """
        super().__init__()

        emb_size=embeddings.shape[1]
        self.item_emb_layer=layers.Embedding(embeddings.shape[0],emb_size,
                        embeddings_initializer=initializers.Constant(embeddings),
                        mask_zero=True,trainable=False,name='item_emb')
        if num_layers<2:
            self.lstm=layers.LSTM(num_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(decay),
                                bias_regularizer=regularizers.l2(decay),
                                activation='sigmoid',name='lstm')
        else:
            self.lstm=[layers.LSTM(num_units, return_sequences=True,
                                kernel_regularizer=regularizers.l2(decay),
                                bias_regularizer=regularizers.l2(decay),
                                activation='sigmoid',name='lstm{}'.format(i))
                                for i in range(num_layers)]

        self.dense_final = layers.Dense(emb_size, kernel_regularizer=regularizers.l2(decay),
                                    bias_regularizer=regularizers.l2(decay),name='dense_final')

    @tf.function
    def call(self, inputs):
        """
        Run forward propagation to produce outputs.

        :param inputs: a tuple of input tensors for predictions.
        :param candidates: a tuple of input tensors for candidates.
        :return: the predicted scores for all candidates.
        """
        # B X T X E
        orders = self.item_emb_layer(inputs)
        mask=self.item_emb_layer.compute_mask(inputs)
        # B X T X U
        if not isinstance(self.lstm, list):
            orders = self.lstm(orders,mask=mask)
        else:
            for lstm in self.lstm:
                orders=lstm(orders,mask=mask)
        # B X U
        out = orders[:, -1, :]
        # out: B x E
        return self.dense_final(out)

# @tf.function
# def Make_temp_rnn(seq_len,embeddings,num_units=32,num_layers=1,decay=0):
#     inputs=Input((seq_len,))
#     emb_size=embeddings.shape[1]
#     output=layers.Embedding(embeddings.shape[0],emb_size,
#                         embeddings_initializer=initializers.Constant(embeddings),
#                         mask_zero=True,trainable=False,name='item_emb')(inputs)
#     if num_layers<2:
#         output=layers.LSTM(num_units, return_sequences=True,
#                             kernel_regularizer=regularizers.l2(decay),
#                             bias_regularizer=regularizers.l2(decay),
#                             activation='sigmoid',name='lstm')(inputs)
#     else:
#         output=inputs
#         for i in range(num_layers):
#             output=layers.LSTM(num_units, return_sequences=True,
#                             kernel_regularizer=regularizers.l2(decay),
#                             bias_regularizer=regularizers.l2(decay),
#                             activation='sigmoid',name='lstm{}'.format(i))(output)
#     output=output[:,-1,:]
#     output=layers.Dense(emb_size, kernel_regularizer=regularizers.l2(decay),
#                                     bias_regularizer=regularizers.l2(decay),name='dense_final')(output)
#     return Model(inputs=inputs,outputs=output)

class RNN2(RNN1):
    """
    RNN 2 model for recommendation, which uses category information.
    """
    def __init__(self, embeddings, num_units=32,
                 num_layers=1, decay=0,categories=None):
        """
        Class initializer.

        :param num_units: the number of hidden units in each LSTM cell.
        :param num_layers: the number of LSTM layers.
        :param decay: an L2 decay parameter for regularization.
        """
        assert categories is not None
        super().__init__(embeddings,num_units, num_layers, decay)

        emb_size=embeddings.shape[1]
        self.cate_emb_layer=layers.Embedding(categories.shape[0],categories.shape[1],
                    embeddings_initializer=initializers.Constant(categories),
                    mask_zero=True,trainable=False,name='cate_emb')
        self.order_dense = layers.Dense(emb_size,name='order_dense')
        self.click_dens=layers.Dense(emb_size,name='click_dense')
        self.softmax=layers.Softmax(1)

    @tf.function
    def call(self, inputs):
        """
        Run forward propagation to produce outputs.

        :param inputs: a tuple of input tensors for predictions.
        :param candidates: a tuple of input tensors for candidates.
        :return: the predicted scores for all candidates.
        """
        
        orders=layers.concatenate([self.item_emb_layer(inputs[0]),self.cate_emb_layer(inputs[0])])
        orders_mask=self.item_emb_layer.compute_mask(inputs[0])
        orders=self.order_dense(orders)
        if self.num_layers<2:
            orders = self.lstm(orders,mask=orders_mask)
        else:
            for layer in self.lstm:
                orders=layer(orders_mask,mask=orders_mask)
        clicks=layers.concatenate([self.item_emb_layer(inputs[1]),self.cate_emb_layer(inputs[1])])
        clicks_mask=self.item_emb_layer.compute_mask(inputs[1])
        clicks = self.click_dense(clicks)
        out = self._run_attention(orders, clicks,clicks_mask)
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
        return layers.average([item_embs, cat_embeddings])

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
    #     return layers.average([self.item_emb.weights[0], cat_dense])

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
        return layers.concatenate([out, out_last], axis=1)


class RNN4(RNN2):
    """
    RNN 3 model for recommendation, which uses clicks for attention keys.
    """

    def __init__(self, embeddings, num_layers, num_units, decay,category):
        """
        Class initializer.

        :param embeddings: embedding vectors of all items.
        :param categories: multi-hot categories of all items.
        :param num_layers: the number of LSTM layers.
        :param num_units: the number of hidden units in each LSTM cell.
        :param decay: an L2 decay parameter for regularization.
        """
        super().__init__(embeddings, num_layers, num_units,decay,category)
        if num_layers<2:
            self.lstm_click=layers.Bidirectional(layers.LSTM(num_units, return_sequences=True,
                                                 kernel_regularizer=regularizers.l2(decay),
                                                 bias_regularizer=regularizers.l2(decay),
                                                 activation='sigmoid',name='lstm_click'))
        else:
            self.lstm_click=[
                layers.Bidirectional(layers.LSTM(num_units, return_sequences=True,
                                                 kernel_regularizer=regularizers.l2(decay),
                                                 bias_regularizer=regularizers.l2(decay),
                                                 activation='sigmoid',name='lstm_click{}'.format(i)))
                            for i in range(num_layers)]
        self.linear_click = layers.Dense(num_units,
                                         kernel_regularizer=regularizers.l2(decay),
                                         bias_regularizer=regularizers.l2(decay),name='linear_click')

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
        return layers.concatenate([out, orders[:, -1, :]], axis=1)
