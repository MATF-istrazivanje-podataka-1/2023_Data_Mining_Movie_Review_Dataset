import os
import codecs
import multiprocessing as mp

import pandas as pd


def read_document(filepath, encoding='utf-8'):
    with codecs.open(filepath, 'r', encoding) as file:
        text = file.read()
    return text


def process_documents(path, labels):
    reviews = {}
    for file in os.listdir(path):
        text = read_document(os.path.join(path, file))
        reviews[text] = labels
    return reviews


def corpus_to_csv(root_directory):
    labels = {'pos': 1, 'neg': 0}
    n_workers = mp.cpu_count()

    process_args = [(os.path.join(root_directory, s, l), labels[l]) for s in ('test', 'train') for l in ('pos', 'neg')]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.starmap(process_documents, process_args)
        reviews = {}
        for review_dict in results:
            reviews.update(review_dict)

    df = pd.DataFrame(list(reviews.items()))
    df = df.sample(frac=1, random_state=42, ignore_index=True)
    df.columns = ['review', 'sentiment']
    df.to_csv('imdb_movie_data.csv', index=False, encoding='utf-8')
