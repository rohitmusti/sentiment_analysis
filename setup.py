import torch
import numpy as np
import spacy
import re
import ujson as json
from statistics import mean
from tqdm import tqdm
from math import ceil, floor
from collections import Counter

from args import get_setup_args
from utils import data_importer, get_logger

def obj_cleaner(obj, counter):
    ret = []
    for line in tqdm(obj):
        new_line = [re.sub('[^a-zA-Z0-9]', '', token.text.lower()) for token in nlp(line) if not token.is_punct]
        for word in new_line:
            counter[word] += 1
        ret.append(new_line)
    return ret

def get_embeddings(emb_file, word_counter):
    embedding_dict= {}

    with open(emb_file, "r", encoding="utf-8") as fh:
        for line in fh:
            array = line.split()
            word = "".join(array[0:-300])
            vector = list(map(float, array[-300:]))
            if word in word_counter:
                embedding_dict[word] = vector
    
    NULL = "--NULL--"
    OOV = "--OOV--"
    word2idx = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    word2idx[NULL] = 0
    word2idx[OOV] = 1

    embedding_dict[NULL] = [0. for _ in range(300)]
    embedding_dict[OOV] = [0. for _ in range(300)]

    idx2emb_dict = {idx: embedding_dict[token] for token, idx in word2idx.items()}
    idx2vec = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]

    return word2idx, idx2vec

def featurize(args, pos_obj, neg_obj, word2idx, idx2vector, percent):
    train_pos_idx_range = (0, floor(percent * len(pos_obj)))
    train_neg_idx_range = (0, floor(percent * len(neg_obj)))
    test_pos_idx_range = (ceil(percent * len(pos_obj)), len(pos_obj))
    test_neg_idx_range = (ceil(percent * len(neg_obj)), len(neg_obj))

    rwtokens_train = []
    sentiment_train = []
    ids = []
    counter = 1
    for line in tqdm(range(train_pos_idx_range[0], train_pos_idx_range[1])):
        new_line = np.zeros([args.review_limit,300])
        for i, word in enumerate(pos_obj[line]):
            # if no vector, randomize
            if word not in word2idx:
                new_line[i] = np.asarray([np.random.normal(scale=0.1) for _ in range(300)])
            else:
                new_line[i] = np.asarray(idx2vector[word2idx[word]])

        ids.append(counter)
        rwtokens_train.append(new_line)
        sentiment_train.append(True)

        counter += 1

    for line in tqdm(range(train_neg_idx_range[0], train_neg_idx_range[1])):
        new_line = np.zeros([args.review_limit,300])
        for i, word in enumerate(pos_obj[line]):
            if word not in word2idx:
                new_line[i] = np.asarray([np.random.normal(scale=0.1) for _ in range(300)])
            else:
                new_line[i] = np.asarray(idx2vector[word2idx[word]])
        rwtokens_train.append(new_line)
        sentiment_train.append(True)

    print(f"created {len(ids)} train examples")
    np.savez(
        args.clean_train_data,
        rwtokens=np.array(rwtokens_train),
        sentiment=np.array(sentiment_train),
        ids=np.array(ids)
    )


    rwtokens_test = []
    sentiment_test = []
    ids = []
    counter = 1
    for line in tqdm(range(test_pos_idx_range[0], test_pos_idx_range[1])):
        new_line = np.zeros([args.review_limit,300])
        for i, word in enumerate(pos_obj[line]):
            if word not in word2idx:
                new_line[i] = np.asarray([np.random.normal(scale=0.1) for _ in range(300)])
            else:
                new_line[i] = np.asarray(idx2vector[word2idx[word]])

        ids.append(counter)
        rwtokens_test.append(new_line)
        sentiment_test.append(True)

        counter += 1

    for line in tqdm(range(test_neg_idx_range[0], test_neg_idx_range[1])):
        new_line = np.zeros([args.review_limit,300])
        for i, word in enumerate(pos_obj[line]):
            if word not in word2idx:
                new_line[i] = np.asarray([np.random.normal(scale=0.1) for _ in range(300)])
            else:
                new_line[i] = np.asarray(idx2vector[word2idx[word]])

        ids.append(counter)
        rwtokens_test.append(new_line)
        sentiment_test.append(True)

        counter += 1
    
    print(f"created {len(ids)} test examples")
    np.savez(
        args.clean_test_data,
        rwtokens=np.array(rwtokens_train),
        sentiment=np.array(sentiment_train),
        ids=np.array(ids)
    )

def main(args):
    logger = get_logger(logging_dir=args.logging_dir, name="data preprocessing")
    word_counter = Counter()

    pos_file_obj = data_importer(args.raw_pos)
    neg_file_obj = data_importer(args.raw_neg)

    logger.info(f"pos file {args.raw_pos} with {len(pos_file_obj)} lines")
    logger.info(f"neg file {args.raw_neg} with {len(neg_file_obj)} lines")

    logger.info(f"tokenizing positive and negative word vectors")
    pos_cleaned = obj_cleaner(pos_file_obj,word_counter)
    neg_cleaned = obj_cleaner(neg_file_obj,word_counter)


    logger.info(f"creating word-to-index and index-to-vector embeddings")
    word2idx, idx2vec = get_embeddings(emb_file=args.word_vecs, word_counter=word_counter)

    logger.info(f"creating training and testing dataset")
    featurize(args, pos_cleaned, neg_cleaned, word2idx, idx2vec, .8)



if __name__ == "__main__":
    args = get_setup_args()
    nlp = spacy.load('en')
    main(args)
    