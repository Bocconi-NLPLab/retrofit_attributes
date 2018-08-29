import argparse
import multiprocessing

import pandas as pd
from gensim.models import Doc2Vec
import sys
import time

from gensim.models.doc2vec import TaggedDocument

parser = argparse.ArgumentParser(description="train DOc2Vec model")
parser.add_argument('--input', help='input file, one text per line')
parser.add_argument('--iters', help='number of iterations', default=1000, type=int)
parser.add_argument('--limit', help='limit number of lines, for test purposes mainly', default=None, type=int)
parser.add_argument('--min_tf', help='minimum term frequency for a word to be counted', default=5, type=int)
parser.add_argument('--prefix', help='save trained model', default=None, type=str, required=True)
parser.add_argument('--subsampling', help='subsampling rate for Doc2Vec, smaller prefers rarer words', default=0.00001,
                    type=float)
parser.add_argument('--window', help='window size for Doc2Vec', default=15, type=int)
args = parser.parse_args()


cores = multiprocessing.cpu_count()
df = pd.read_csv(args.input, sep='\t')

corpus = [TaggedDocument(words=line.split(' '), tags=['ID%08d' % line_no]) for line_no, line in enumerate(df.text)]

print('\nbuilding model', file=sys.stderr, flush=True)
model = Doc2Vec(vector_size=300, window=args.window, min_count=args.min_tf, negative=5, hs=0,
                     workers=cores, epochs=args.iters, sample=args.subsampling, dm=0, dbow_words=0)

model.build_vocab(corpus)

print('\ntraining model', file=sys.stderr, flush=True)
# account for the different signatures in Python 3.5 and 3.4
start = time.time()
try:
    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
except TypeError:
    model.train(corpus)
print('... done in %.2f!' % (time.time() - start), file=sys.stderr, flush=True)

# remove unneeded memory stuff
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

model.save('%s.model' % args.prefix)
print('\nmodel saved as %s.model' % args.prefix, file=sys.stderr, flush=True)
