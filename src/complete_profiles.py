from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import numpy as np
import sys
import pandas as pd
from gensim.models import Doc2Vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser(description="complete demographic profles in a sample based on their embeddings")
parser.add_argument('--target', help='demographic variable to predict', choices={'age', 'gender'}, type=str, required=True)
parser.add_argument('--model', help='Doc2Vec model', type=str, required=True)
parser.add_argument('--data', help='input data with target variables', type=str, required=True)
parser.add_argument('--limit', help='limit instances for test purposes', default=None, type=int)
parser.add_argument('--runs', help='number of runs to average over', default=10, type=int)
parser.add_argument('--alpha', help='weight of original vector (0-1.0). Default 0.5. alpha=1 is the same as no retrofitting', default=0.5, type=float)
parser.add_argument('--size', help='number of assumed known profiles', default=None, type=int, required=True)
parser.add_argument('--prefix', help='output prefix', type=str, required=True)

args = parser.parse_args()


def retrofit(vectors, neighbors, normalize=False, num_iters=10, alpha=0.5):
    N, D = vectors.shape

    new_vectors = deepcopy(vectors)

    if normalize:
        for c in range(N):
            vectors[c] /= np.sqrt((vectors[c]**2).sum() + 1e-6)

    # run retrofitting
    print("retrofitting", file=sys.stderr, flush=True)

    beta = 1 - alpha

    for it in range(num_iters):
        print("\t{}:".format(it + 1), end=' ', file=sys.stderr, flush=True)
        # loop through every city
        for c in range(N):
            print(".", end='', file=sys.stderr, flush=True)
            instance_neighbours = neighbors[c]
            num_neighbours = len(instance_neighbours)
            #no neighbours, pass - use data estimate
            if num_neighbours == 0:
                continue

            # the weight of the data estimate is the number of neighbours, plus the sum of all neighboring vectors, normalized
            new_vectors[c] = (alpha * (num_neighbours * vectors[c]) + beta * (new_vectors[instance_neighbours].sum(axis=0))) / (alpha * num_neighbours + beta * num_neighbours)

        print('', file=sys.stderr)
    return new_vectors

def get_neighbors(y):
    value2id = {value: {idx for idx, v2 in enumerate(y) if v2 == value} for value in y}
    id2neighbors = {}
    for _, idx in value2id.items():
        for id_ in idx:
            id2neighbors[id_] = list(idx)
            id2neighbors[id_].remove(id_)
    return id2neighbors


model = Doc2Vec.load(args.model)
profile_vectors = np.array([model.docvecs[profile] for profile in model.docvecs.doctags])
num_profiles, num_dims = profile_vectors.shape
df = pd.read_csv(args.data, sep='\t')

run_results = []

labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(df[args.target])


for run in range(args.runs):
    print('RUN {}'.format(run+1), file=sys.stderr, flush=True)
    print('*' * 10, file=sys.stderr, flush=True)

    # select subset of profiles to train on
    subset_ids = np.random.choice(num_profiles, size=args.size, replace=False)
    X = profile_vectors[subset_ids]
    # labels = df[args.target].values
    y = labels[subset_ids]

    # vectorize text data
    print('\tvectorize text data', file=sys.stderr, flush=True)
    vectorizer = DictVectorizer()
    transformer = TfidfTransformer()
    X_bow_count = vectorizer.fit_transform(df.text[subset_ids].apply(lambda x: Counter(x.split(' '))))
    X_bow = transformer.fit_transform(X_bow_count)
    selector = SelectKBest(chi2, k=num_dims).fit(X_bow, y)
    X_bow = selector.transform(X_bow)

    print('\tretrofitting', file=sys.stderr, flush=True)
    # get neighbor dictionary
    D = get_neighbors(y)
    # retrofit training embeddings
    X_retrofit = retrofit(X, D, normalize=False, num_iters=10, alpha=args.alpha)
    X_bow_retrofit = retrofit(X_bow.todense(), D, normalize=False, num_iters=10, alpha=args.alpha) + 0.0000001

    # print(type(X), type(X_retrofit), type(X_bow), type(X_bow_retrofit))
    # sys.exit()
    # train different models
    clf_embeddings = LogisticRegression()
    clf_retrofit_embeddings = LogisticRegression()
    clf_bow = LogisticRegression()
    clf_bow_retrofit = LogisticRegression(max_iter=2, verbose=3)

    print('\ttraining embedding model', file=sys.stderr, flush=True)
    clf_embeddings.fit(X, y)
    print('\ttraining retrofit embedding model', file=sys.stderr, flush=True)
    clf_retrofit_embeddings.fit(X_retrofit, y)
    print('\ttraining BOW model', file=sys.stderr, flush=True)
    clf_bow.fit(X_bow, y)
    print('\ttraining retrofit BOW model', file=sys.stderr, flush=True)
    clf_bow_retrofit.fit(X_bow_retrofit, y)

    # test data
    mask = np.ones(num_profiles, np.bool)
    mask[subset_ids] = 0
    Z = profile_vectors[mask]
    gold = labels[mask]
    # vectorize test data
    Z_bow = selector.transform(vectorizer.transform(df.text[mask].apply(lambda x: Counter(x.split(' ')))))

    print('\tfinding translation matrices', file=sys.stderr, flush=True)
    translation_matrix_embeddings = np.linalg.lstsq(X, X_retrofit, rcond=None)[0]
    translation_matrix_bow = np.linalg.lstsq(X_bow.todense(), X_bow_retrofit, rcond=None)[0]

    Z_retrofit = np.dot(Z, translation_matrix_embeddings)
    Z_bow_retrofit = np.dot(Z_bow, translation_matrix_bow)

    print('\tpredicting', file=sys.stderr, flush=True)
    # predictions
    predictions_embeddings = clf_embeddings.predict(Z)
    predictions_retrofit_embeddings = clf_retrofit_embeddings.predict(Z_retrofit)
    predictions_bow = clf_bow.predict(Z_bow)
    predictions_bow_retrofit = clf_bow_retrofit.predict(Z_bow_retrofit)

    print('\tscoring', file=sys.stderr, flush=True)
    f1_embeddings = f1_score(gold, predictions_embeddings, average='micro')
    f1_retrofit_embeddings = f1_score(gold, predictions_retrofit_embeddings, average='micro')
    f1_bow = f1_score(gold, predictions_bow, average='micro')
    f1_bow_retrofit = f1_score(gold, predictions_bow_retrofit, average='micro')

    outcomes = [f1_embeddings, f1_retrofit_embeddings, f1_bow, f1_bow_retrofit]
    print('\t{}'.format(outcomes))
    run_results.append(outcomes)

run_results = [[args.size] + np.array(run_results).mean(axis=0).tolist()]
results = pd.DataFrame(run_results, columns=['known', 'embeddings', 'retrofit', 'BOW', 'BOW_retrofit'])
results.to_csv('{}{}_{}.csv'.format(args.prefix, args.target, args.size))
