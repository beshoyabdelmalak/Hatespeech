import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from datasets import load_dataset

from sklearn.model_selection import train_test_split


class Preprocessing:
    def __init__(self, args):
        self.max_len = args.max_len
        self.max_words = args.max_words

        self.train_path = os.path.join(os.path.dirname(__file__), "../data/train.csv")
        self.test_path = os.path.join(os.path.dirname(__file__), "../data/test.csv")
        self.valid_path = os.path.join(os.path.dirname(__file__), "../data/valid.csv")

    def load_data(self):
        df_train = pd.read_csv(self.train_path)
        df_test = pd.read_csv(self.test_path)
        df_valid = pd.read_csv(self.valid_path)

        self.x_train = df_train["text"].values
        self.y_train = df_train["chosen_label"].values

        self.x_test = df_test["text"].values
        self.y_test = df_test["chosen_label"].values

        self.x_valid = df_valid["text"].values
        self.y_valid = df_valid["chosen_label"].values

    def prepare_tokens(self):
        self.tokens = Tokenizer(num_words=self.max_words)
        self.tokens.fit_on_texts(self.x_train)
        word_index = self.tokens.word_index
        print("Found %s unique tokens." % len(word_index))

    def sequence_to_token(self, x):
        sequences = self.tokens.texts_to_sequences(x)
        return sequence.pad_sequences(sequences, maxlen=self.max_len)
