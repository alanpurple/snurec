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
from tensorflow.keras import Model
import numpy as np
from . import models
from utils import read_instances,read_titles,initialize_model

@click.command()
@click.option('--algorithm', '-a', type=str, default=None)
@click.option('--load', '-l', type=click.Path(), default=None)
@click.option('--data', type=click.Path(), default='../out/instances')
@click.option('--gpu', type=int, default=0)
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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    path_items = os.path.join(data, 'items')
    path_instances = os.path.join(data, 'test')
    path_out = os.path.join(load, 'predictions')

    titles = read_titles(path_items)
    item_emb=np.load('../doc2vec_128_2_10epochs_table.npy')
    num_items=item_emb.shape[0]-1
    emb_len=item_emb.shape[1]
    dataset = read_instances(path_instances, batch_size=128)

    # model = initialize_model(algorithm, *candidates)
    # if model.is_trainable:
    #     model.load_weights(os.path.join(load, 'model/model'))
    if algorithm[:3]=='rnn':
        model = Model.load(os.path.join(load, 'model/model.tf'))
    elif algorithm in {'last', 'average'}:
        model = models.BaselineModel(item_emb, mode=algorithm)
    else:
        raise Exception('unknown algorithm')

    p_counts, n_counts = 0, 0
    for inputs, labels in dataset:
        if algorithm=='rnn-v2' or algorithm=='rnn-v3' or algorithm=='rnn-v4':
             scores, predictions = tf.math.top_k(model(inputs),top_k,sorted=True)
        else:
            scores, predictions = tf.math.top_k(model(inputs),top_k,sorted=True)
        orders = inputs[1].numpy()
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
