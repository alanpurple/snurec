""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

refactored by Alan Anderson (alan@wemakeprice.com)

File: rec/eval.py
- Evaluate a trained recommendation model.

Version: 1.0.0

rewritten by Alan Anderson (alan@wemakeprice.com)

"""
import click
import os
import tensorflow as tf
from tensorflow_core.python.keras import layers,initializers
import numpy as np
import pickle
import models
from utils import read_instances,read_titles,initialize_model

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

@click.command()
@click.option('--algorithm', '-a', type=str, default=None)
@click.option('--load', '-l', type=click.Path(), default=None)
@click.option('--data', type=click.Path(), default='../out/instances')
@click.option('--pos-cases', type=int, default=100)
@click.option('--neg-cases', type=int, default=10000)
@click.option('--top-k', type=int, default=10)
def main(algorithm=None, load=None, data='../out/instances', gpu=0, pos_cases=100, neg_cases=10000, top_k=10):
    """
    Evaluate a trained recommendation model.

    :param algorithm: the name of an algorithm to use.
    :param load: the path to load the trained model.
    :param data: generated instances for the training.
    :param gpu: a list of GPUs to use in the training.
    :param pos_cases: the number of positive cases to store.
    :param neg_cases: the number of negative cases to store.
    :param top_k: the number of items for evaluations.
    :return: None.
    """
    assert algorithm in {'last', 'average'} or algorithm[:3]=='rnn'
    assert load is not None

    path_items = os.path.join(data, 'items')
    path_instances = os.path.join(data, 'test')
    path_out = os.path.join(load, 'predictions')

    titles = read_titles(path_items)
    item_emb=np.load('doc2vec_32_2_10epochs_table.npy')
    emb_size=item_emb.shape[1]
    has_cate=False
    if algorithm=='rnn-v2' or algorithm=='rnn-v3' or algorithm=='rnn-v4':
        has_cate=True
        category_table=np.load('./cate.npy')
    dataset = read_instances(path_instances, batch_size=128)

    @tf.function
    def get_baseline_output(inputs,mode):
        orders=layers.Embedding(item_emb.shape[0],emb_size,
                        embeddings_initializer=initializers.Constant(item_emb),
                        mask_zero=True,trainable=False,name='item_emb')(inputs)

        if mode == 'average':
            out = tf.reduce_sum(orders, axis=1)
            out /= tf.expand_dims(get_sequence_length(orders), axis=1)
        elif mode == 'last':
            out = orders[:, -1, :]
        else:
            raise ValueError(mode)
        return out

    is_baseline=False
    # To Do: insert mechanism to load num_layer,num_units and decay settings
    if algorithm[:3]=='rnn':
        if has_cate:
            model=initialize_model(algorithm,item_emb,2,32,0,category_table)
            
        else:
            model=initialize_model(algorithm,item_emb,2,32,0)
        with open('model/best_weights.pkl','rb') as best_temp:
            weight_dict=pickle.load(best_temp)
        for k,v in weight_dict.items():
            for layer in model.layers:
                if layer.name==k:
                    layer.set_weights(v)
    elif algorithm in {'last', 'average'}:
        is_baseline=True
    else:
        raise Exception('unknown algorithm')

    p_counts, n_counts = 0, 0
    for inputs, labels in dataset:
        if is_baseline:
            logits=get_baseline_output(inputs,algorithm)
        else:
            logits=model(inputs)
        output=tf.matmul(logits,tf.transpose(item_emb))
        scores, predictions = tf.math.top_k(output,top_k,sorted=True)
        orders = inputs[1].numpy() if has_cate else inputs.numpy()
        labels = labels.numpy()
        scores = scores.numpy()
        predictions = predictions.numpy()

        for i in range(labels.shape[0]):
            if np.sum(predictions[i, :] == labels[i]) > 0:
                if p_counts >= pos_cases:
                    continue
                p_counts += 1
                path_file = os.path.join(
                    path_out, 'positives', f'case-{p_counts}.txt')
            else:
                if n_counts >= neg_cases:
                    continue
                n_counts += 1
                path_file = os.path.join(
                    path_out, 'negatives', f'case-{n_counts}.txt')

            str_items = titles[orders[i, orders[i, :] >= 0]]
            str_preds = titles[predictions[i, :]]
            str_label = titles[labels[i]]

            os.makedirs(os.path.dirname(path_file), exist_ok=True)
            with open(path_file, 'w') as f:
                for j, obs in enumerate(str_items):
                    f.write(f'[ORDER {j + 1:02d}] {obs}\n')
                f.write('\n')
                f.write(f'[ANSWER] {str_label}\n\n')
                for score, name in zip(scores[i, :], str_preds):
                    f.write(f'[{score:6.3f}] {name}\n')

        if p_counts > pos_cases and n_counts > neg_cases:
            break


if __name__ == '__main__':
    main()
