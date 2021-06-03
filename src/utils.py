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

		# check if csv files exist
		if not os.path.exists(self.train_path):
			self.dataset = load_dataset("hatexplain")
			self.create_csv_files()


	def create_csv_files(self):
		training_posts = self.dataset["train"].to_pandas()
		test_posts = self.dataset["test"].to_pandas()
		validation_posts = self.dataset["validation"].to_pandas()

        # Prepare columns
		training_posts["chosen_label"] = training_posts["annotators"].apply(
            self.get_annotator_label
        )
		training_posts["text"] = training_posts["post_tokens"].apply(self.clean_text)
		training_posts = training_posts[["chosen_label", "text"]]

		test_posts["chosen_label"] = test_posts["annotators"].apply(
            self.get_annotator_label
        )
		test_posts["text"] = test_posts["post_tokens"].apply(self.clean_text)
		test_posts = test_posts[["chosen_label", "text"]]

		validation_posts["chosen_label"] = validation_posts["annotators"].apply(
            self.get_annotator_label
        )
		validation_posts["text"] = validation_posts["post_tokens"].apply(
            self.clean_text
        )
		validation_posts = validation_posts[["chosen_label", "text"]]

		training_posts.to_csv(self.train_path, index=False)
		test_posts.to_csv(self.test_path, index=False)
		validation_posts.to_csv(self.valid_path, index=False)


	def clear_emojis(self, text):
		return re.sub(r"\W+", " ", text)


	def remove_special_words(self, text):
		return re.sub(r"\<.*?\>", " ", text)


	def trim_string(self, text):
		return " ".join(text[: 200])


	def clean_text(self, text):
		trimed = self.trim_string(text)
		cleaned_text = self.remove_special_words(trimed)
		cleaned_text = self.clear_emojis(cleaned_text)
		return cleaned_text


	def get_annotator_label(self, annotators):
		labels = [annotators["label"][0], annotators["label"][1], annotators["label"][2]]
		counter = Counter(labels)
		majority_label = counter.most_common(1)[0][0]
        # for binary classifier
        # labels = {0: hatespeech, 1:normal, 2:offensive}
		if majority_label == 1:
			return 0
		return 1
        # return majority_label


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
		print('Found %s unique tokens.' % len(word_index))


	def sequence_to_token(self, x):
		sequences = self.tokens.texts_to_sequences(x)
		return sequence.pad_sequences(sequences, maxlen=self.max_len)