""" Wemakeprice Recommendation Project.

Authors:
- Hyunsik Jeon (jeon185@snu.ac.kr)
- Jaemin Yoo (jaeminyoo@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

File: rec/title.py
- Learn the embedding vectors of all items using their titles.

Version: 1.0.0

"""
import random

import click
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from konlpy.tag import Okt

from rec.utils import *


def to_documents(df):
    """
    Convert a DataFrame to a list of documents.

    :param df: the DataFrame.
    :return: the list of documents.
    """
    parser = Okt()

    def func(row):
        return TaggedDocument(parser.nouns(row[PROD_NM]), [row[PROD_NO]])

    return df.apply(func, axis=1).tolist()


def train_model(documents, vector_size, window):
    """
    Train a Doc2Vec model using training documents.

    :param documents: a list of documents to train the model.
    :param vector_size: the size of embedding vectors.
    :param window: the window size for training the model.
    :return: the trained models.
    """
    model = Doc2Vec(vector_size=vector_size,
                    window=window,
                    workers=8,
                    min_count=1,
                    alpha=0.025,
                    min_alpha=0.00025,
                    epochs=5)
    model.build_vocab(documents)
    model.train(documents=documents,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    return model


def to_vectors(model, documents):
    """
    Convert documents into vectors using a language model.

    :param model: the trained language model.
    :param documents: documents to convert to vectors.
    :return: the generated vectors.
    """
    vectors = []
    for doc in documents:
        tag = doc.tags[0]
        if tag in model.docvecs:
            v = model.docvecs[tag]
        else:
            v = model.infer_vector(doc.words)
        vectors.append(v)
    return vectors


def run_test(model, df):
    """
    Test a trained language model for evaluation.

    :param model: the trained language model.
    :param df: a DataFrame containing documents.
    :return: None.
    """
    names = df[PROD_NM]
    ids = df[PROD_NO]

    rand = random.randint(0, len(ids))
    sims = model.docvecs.most_similar([model.docvecs[rand]], topn=11)
    for (i, _) in sims:
        print(names[i])


@click.command()
@click.option('--data', type=click.Path(exists=True), default='../data/product.csv')
@click.option('--out', type=click.Path(), default='../out/title')
@click.option('--load', type=click.Path(), default=None)
@click.option('--test', is_flag=True)
@click.option('--vector-size', type=int, default=64)
@click.option('--window', type=int, default=2)
def main(data='../data/product.csv', out='../out/title', load=None, test=True, vector_size=64, window=2):
    """
    Train a language model using the titles of all items.

    :param data: the path to the titles of items.
    :param out: the path to store the trained model.
    :param load: whether to load an existing model or not.
    :param test: whether to test the trained model or not.
    :param vector_size: the size of embedding vectors.
    :param window: the window size when training the model.
    :return: None.
    """
    df = pd.read_csv(data, dtype={PROD_NO: str})
    documents = to_documents(df)
    os.makedirs(out, exist_ok=True)

    model_path = f'{out}/saved_model'
    if load is None:
        model = train_model(documents, vector_size, window)
        model.save(model_path)
    else:
        model = Doc2Vec.load(load)

    id_path = f'{out}/ids.tsv'
    title_path = f'{out}/titles.tsv'
    df[PROD_NO].to_csv(id_path, header=False, index=False)
    df[PROD_NM].to_csv(title_path, header=False, index=False)

    vector_path = f'{out}/vectors.tsv'
    vectors = to_vectors(model, documents)
    np.savetxt(vector_path, vectors, delimiter='\t')

    if test:
        run_test(model, df)


if __name__ == '__main__':
    main()
