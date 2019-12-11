""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

refactored by Alan Anderson (alan@wemakeprice.com)

File: rec/train.py
- Train a recommendation model and save it.

Version: 1.0.0

"""
import time

import click
import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers,losses
from rec.utils import *


def count_batches(dataset):
    """
    Count the number of batches in a dataset.

    :param dataset: the dataset.
    :return: the number of batches.
    """
    count = 0
    for _ in dataset:
        count += 1
    return count

@tf.function
def evaluate_loss(model, data):
    """
    Evaluate the current loss of a given model for all data.

    :param model: the model to be evaluated.
    :param data: the dataset of multiple batches.
    :return: the calculated loss.
    """
    loss, batches = 0, 0
    for inputs, labels in data:
        logits = model(inputs, (model.embeddings, model.categories))
        cce=losses.SparseCategoricalCrossentropy()
        loss+= tf.reduce_mean(cce(labels, logits))
        batches+=1
    return loss / batches


def evaluate_accuracy(has_cate,model, data, embeddings, k,categories=None):
    """
    Evaluate the accuracy of a given model for all data.

    :param model: the model to be evaluated.
    :param data: the dataset of multiple batches.
    :param embeddings: the embedding vectors of candidates.
    :param categories: the multi-hot categories of candidates.
    :param top_k: the number of items to be evaluated.
    :return: the calculated accuracy.
    """
    n_data, n_corrects = 0, 0
    for inputs, labels in data:
        if has_cate:
            predictions = model(inputs, (embeddings,categories))
        else:
            predictions = model(inputs,embeddings)
        top_k= tf.math.top_k(predictions, k, sorted=True)[1]
        compared = tf.equal(tf.expand_dims(labels, axis=1), top_k)
        corrects = tf.reduce_sum(tf.cast(compared, dtype=tf.float32), axis=1)
        accuracy = tf.reduce_mean(corrects)
        n_data += labels.shape[0]
        n_corrects += accuracy.numpy().item() * labels.shape[0]
    return n_corrects * 100 / n_data


@click.command()
@click.option('--algorithm', '-a', type=str, default=None)
@click.option('--data', type=click.Path(), default='../out/instances')
@click.option('--top-k', type=int, default=10)
@click.option('--num-epochs', type=int, default=1000)
@click.option('--num-units', type=int, default=None)
@click.option('--num-layers', type=int, default=None)
@click.option('--emb-way', type=str, default=None)
@click.option('--lr', type=float, default=1e-3)
@click.option('--decay', type=float, default=1e-3)
@click.option('--batch-size', type=int, default=256)
@click.option('--patience', type=int, default=2)  # 10 is recommended for 2W.
@click.option('--gpu', type=int, default=0)
@click.option('--out', type=str, default=None)
def main(data='../out/instances', algorithm=None, top_k=100, lr=1e-3, decay=1e-3,
         num_epochs=1000, num_units=None, num_layers=None,
         emb_way=None, batch_size=256, patience=2, gpu=0, out=None):
    """
    Train a recommendation model.

    :param data: generated instances for the training.
    :param algorithm: the name of an algorithm to use.
    :param top_k: the number of items for evaluations.
    :param lr: a learning rate for the training.
    :param decay: an L2 decay parameter for regularization.
    :param num_epochs: the number of training epochs.
    :param num_units: the number of units in LSTM cells.
    :param num_layers: the number of LSTM layers.
    :param emb_way: how to generate embedding vectors of items.
    :param batch_size: a batch size.
    :param patience: the number of epochs to wait until the termination.
    :param gpu: a list of GPUs to use in the training.
    :param out: the path to store the outputs.
    :return: None.
    """
    assert algorithm[:3]=='rnn'

    start_time = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    users = read_users(os.path.join(data, 'users'))
    item_emb=np.load('../doc2vec_128_2_10epochs_table.npy')
    num_items=item_emb.shape[0]-1
    emb_len=item_emb.shape[1]
    has_cate=False
    if algorithm=='rnn-v2' or algorithm=='rnn-v3' or algorithm=='rnn-v4':
        has_cate=True
        category_table=np.load('./cate.npy')

    trn_path = os.path.join(data, 'training')
    val_path = os.path.join(data, 'validation')
    test_path = os.path.join(data, 'test')

    trn_data = read_instances(trn_path, batch_size,is_exp=has_cate)
    val_data = read_instances(val_path, batch_size,is_exp=has_cate)
    test_data = read_instances(test_path, batch_size,is_exp=has_cate)

    num_users = users.shape[0]
    assert num_users is not None
    if emb_way is None:
        emb_way='mean'
    if has_cate:
        model = initialize_model(
            algorithm,
            num_items,emb_len,item_emb,
            category_table,
            num_layers,num_units,
            decay=decay,
            emb_way=emb_way)
    else:
        model = initialize_model(
            algorithm,
            num_items,emb_len,item_emb,
            num_layers,num_units,
            decay=decay,
            emb_way=emb_way)

    out = f'../out/{algorithm}' if out is None else out
    os.makedirs(out, exist_ok=True)
    out_loss = os.path.join(out, 'losses.tsv')
    if os.path.exists(out_loss):
        os.remove(out_loss)

    trn_batches = count_batches(trn_data)
    best_epoch = 0
    best_loss = np.inf
    os.makedirs(os.path.join(out, 'model'), exist_ok=True)
    optimizer = optimizers.Adam(learning_rate=lr)

    cce=losses.SparseCategoricalCrossentropy()

    for epoch in range(num_epochs + 1):
        if epoch == 0:
            trn_loss = evaluate_loss(model, trn_data)
        else:
            desc = f'Epoch {epoch}'
            trn_loss = 0
            for inputs, labels in tqdm.tqdm(trn_data, desc, trn_batches):
                with tf.GradientTape() as tape:
                    if has_cate:
                        logits = model(inputs, (model.embeddings, model.categories))
                    else:
                        logits = model(inputs,model.embeddings)
                    loss = tf.reduce_mean(cce(labels, logits))
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                trn_loss+=loss
            trn_loss /= trn_batches

        val_loss = evaluate_loss(model, val_data)
        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            model.save_weights(os.path.join(out, 'model/model'))

        with open(out_loss, 'a') as f:
            f.write(f'{epoch:4d}\t{trn_loss:10.4f}\t{val_loss:10.4f}\t')
            if best_epoch == epoch:
                f.write('BEST')
            f.write('\n')

        if epoch >= best_epoch + patience:
            break

    model.load_weights(os.path.join(out, 'model/model'))
    model.save(os.path.join(out, 'model/model.tf'))

    trn_loss = evaluate_loss(model, trn_data)
    val_loss = evaluate_loss(model, val_data)

    if has_cate:
        trn_acc = evaluate_accuracy(True,model, trn_data, item_emb,top_k,category_table)
        val_acc = evaluate_accuracy(True,model, val_data, item_emb,top_k,category_table)
        test_acc = evaluate_accuracy(True,model, test_data, item_emb,top_k,category_table)
    else:
        trn_acc = evaluate_accuracy(False,model, trn_data, item_emb,top_k)
        val_acc = evaluate_accuracy(False,model, val_data, item_emb,top_k)
        test_acc = evaluate_accuracy(False,model, test_data, item_emb,top_k)

    out_summary = os.path.join(out, 'summary.tsv')

    with open(out_summary, 'w') as f:
        f.write(f'{best_epoch:4d}\t')
        f.write(f'{trn_loss:.4f}\t')
        f.write(f'{val_loss:.4f}\t')
        f.write(f'{trn_acc:.4f}\t')
        f.write(f'{val_acc:.4f}\t')
        f.write(f'{test_acc:.4f}\t')
        f.write(f'{time.time() - start_time}\n')


if __name__ == '__main__':
    main()
