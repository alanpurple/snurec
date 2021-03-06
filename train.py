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

rewritten by Alan Anderson (alan@wemakeprice.com)

"""
import time
from functools import partial
import os
import click
import tqdm
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses,layers,initializers,optimizers
from utils import read_instances,read_titles,read_users,initialize_model

def gen(df):
    for _,row in df.iterrows():
        yield row.x[:,0],row.label


@click.command()
@click.option('--algorithm', '-a', type=str, default='rnn-v1')
@click.option('--data', type=click.Path(), default='/mnt/sda1/common/SNU_recommendation/wmind_data/ver2')
@click.option('--top-k', type=int, default=10)
@click.option('--num-epochs', type=int, default=10)
@click.option('--num-units', type=int, default=32)
@click.option('--num-layers', type=int, default=2)
@click.option('--lr', type=float, default=1e-3)
@click.option('--decay', type=float, default=1e-3)
@click.option('--batch-size', type=int, default=256)
@click.option('--patience', type=int, default=2)  # 10 is recommended for 2W.
@click.option('--out', type=str, default='.')
def main(data='/mnt/sda1/common/SNU_recommendation/wmind_data/ver2',
         algorithm='rnn-v1', top_k=100, lr=1e-3, decay=1e-3,
         num_epochs=10, num_units=32, num_layers=2,
         batch_size=256, patience=2, out='.'):
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
    :param batch_size: a batch size.
    :param patience: the number of epochs to wait until the termination.
    :param out: the path to store the outputs.
    :return: None.
    """
    assert algorithm[:3]=='rnn'

    start_time = time.time()

    #users = read_users(os.path.join(data, 'users'))
    item_emb=np.load('doc2vec_128_2_10epochs_table.npy').astype(np.float32)
    has_cate=False
    if algorithm=='rnn-v2' or algorithm=='rnn-v3' or algorithm=='rnn-v4':
        has_cate=True
        category_table=np.load('./cate.npy').astype(np.float32)

    # trn_path = os.path.join(data, 'training')
    # val_path = os.path.join(data, 'validation')
    # test_path = os.path.join(data, 'test')

    # trn_data = read_instances(trn_path, batch_size,is_exp=has_cate)
    # val_data = read_instances(val_path, batch_size,is_exp=has_cate)
    # test_data = read_instances(test_path, batch_size,is_exp=has_cate)
    train_df=pd.read_pickle(data+'/train_data_df.pkl.gz','gzip')

    val_df=pd.read_pickle(data+'/valid_data_df.pkl.gz','gzip')
    test_df=pd.read_pickle(data+'/test_data_df.pkl.gz','gzip')

    seq_len=train_df.apply(lambda row:len(row.x),axis=1).max()

    trn_data=tf.data.Dataset.from_generator(partial(gen,train_df),
            (tf.int32,tf.int32),((None,),())).shuffle(512).padded_batch(batch_size,([seq_len],[]),drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    # NoneType not supported error occured without "drop_remainder=True" setting, in line "n_data+=label.shape[0]" in evaluate_accuracy
    val_data=tf.data.Dataset.from_generator(partial(gen,val_df),
            (tf.int32,tf.int32),((None,),())).padded_batch(64,([seq_len],[]),drop_remainder=True)
    test_data=tf.data.Dataset.from_generator(partial(gen,test_df),
            (tf.int32,tf.int32),((None,),())).padded_batch(64,([seq_len],[]),drop_remainder=True)

    print('dataset loading complete')

    cce=losses.SparseCategoricalCrossentropy(True)
    @tf.function
    def evaluate_loss(model, data):
        """
        Evaluate the current loss of a given model for all data.

        :param model: the model to be evaluated.
        :param data: the dataset of multiple batches.
        :return: the calculated loss.
        """
        loss=np.float32(0.)
        batches=np.float32(0.)
        for inputs, labels in data:
            logits = model(inputs)
            logits=tf.matmul(logits,tf.transpose(item_emb))
            loss+= tf.reduce_mean(cce(labels, logits))
            batches+=1.
        return loss / batches

    @tf.function
    def evaluate_accuracy(has_cate,model, data, k):
        """
        Evaluate the accuracy of a given model for all data.

        :param model: the model to be evaluated.
        :param data: the dataset of multiple batches.
        :param embeddings: the embedding vectors of candidates.
        :param categories: the multi-hot categories of candidates.
        :param top_k: the number of items to be evaluated.
        :return: the calculated accuracy.
        """
        n_data, n_corrects = 0., 0.
        for inputs, labels in data:
            logits = model(inputs)
            logits=tf.matmul(logits,tf.transpose(item_emb))
            top_k= tf.math.top_k(logits, k, sorted=True)[1]
            compared = tf.equal(tf.expand_dims(labels, axis=1), top_k)
            corrects = tf.reduce_sum(tf.cast(compared, tf.float32), axis=1)
            accuracy = tf.reduce_mean(corrects)
            n_data += labels.shape[0]
            n_corrects += accuracy * labels.shape[0]
        return n_corrects * 100 / n_data

    if has_cate:
        model=initialize_model(
        algorithm,item_emb,
        num_layers,num_units,
        decay,category_table)
    else:
        model = initialize_model(
        algorithm,item_emb,
        num_layers,num_units,
        decay)

    out = f'../out/{algorithm}' if out is None else out
    os.makedirs(out, exist_ok=True)
    out_loss = os.path.join(out, 'losses.tsv')
    if os.path.exists(out_loss):
        os.remove(out_loss)

    best_epoch = 0
    best_loss = np.inf
    os.makedirs(os.path.join(out, 'model'), exist_ok=True)
    optimizer = optimizers.Adam(learning_rate=lr)

    num_batches=np.float64(int(len(train_df)/batch_size))
    
    @tf.function
    def compute_loss(inputs):
        logits=model(inputs)
        logits=tf.matmul(logits,tf.transpose(item_emb))
        return tf.reduce_mean(cce(labels, logits))
    for epoch in range(num_epochs + 1):
        if epoch == 0:
            trn_loss = evaluate_loss(model, trn_data)
        else:
            print('training for epoch{}'.format(epoch))
            desc = f'Epoch {epoch}'
            trn_loss = 0.
            for inputs, labels in tqdm.tqdm(trn_data, desc, num_batches):
                with tf.GradientTape() as tape:
                    loss=compute_loss(inputs)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                trn_loss+=loss
            trn_loss /= num_batches

        val_loss = evaluate_loss(model, val_data)
        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_weights_dict={}
            for layer in model.layers:
                if has_cate:
                    if layer.name!='item_emb' and layer.name!='cate_emb':
                        best_weights_dict[layer.name]=layer.get_weights()
            with open('model/best_weights.pkl','wb') as best_temp:
                pickle.dump(best_weights_dict,best_temp)
            with open('model/epoch{}.pkl'.format(epoch),'wb') as epoch_save:
                pickle.dump(best_weights_dict,epoch_save)


        with open(out_loss, 'a') as f:
            f.write(f'{epoch:4d}\t{trn_loss:10.4f}\t{val_loss:10.4f}\t')
            if best_epoch == epoch:
                f.write('BEST')
            f.write('\n')

        if epoch >= best_epoch + patience:
            break
    
    for k,v in best_weights_dict:
        for layer in model.layers:
            if layer.name==k:
                layer.set_weights(v)

    trn_loss = evaluate_loss(model, trn_data)
    val_loss = evaluate_loss(model, val_data)

    trn_acc = evaluate_accuracy(has_cate,model, trn_data,top_k)
    val_acc = evaluate_accuracy(has_cate,model, val_data,top_k)
    test_acc = evaluate_accuracy(has_cate,model, test_data,top_k)

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
