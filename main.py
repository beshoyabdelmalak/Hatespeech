import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


from src import Preprocessing
from src import LSTM

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src import parameter_parser


class DatasetMaper(Dataset):
	'''
	Handles batches of dataset
	'''
	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
		

class Execute:
	'''
	Class for execution. Initializes the preprocessing as well as the 
	LSTM model
	'''

	def __init__(self, args):
		self.__init_data__(args)
		
		self.args = args
		self.batch_size = args.batch_size
		
		self.model = LSTM(args)
		
	def __init_data__(self, args):
		'''
		Initialize preprocessing from raw dataset to dataset split into training and testing
		Training and test datasets are index strings that refer to tokens
		'''
		self.preprocessing = Preprocessing(args)
		self.preprocessing.load_data()
		self.preprocessing.prepare_tokens()

		raw_x_train = self.preprocessing.x_train
		raw_x_test = self.preprocessing.x_test
		raw_x_valid = self.preprocessing.x_valid

		self.y_train = self.preprocessing.y_train
		self.y_test = self.preprocessing.y_test
		self.y_valid = self.preprocessing.y_valid

		self.x_train = self.preprocessing.sequence_to_token(raw_x_train)
		self.x_test = self.preprocessing.sequence_to_token(raw_x_test)
		self.x_valid = self.preprocessing.sequence_to_token(raw_x_valid)

		
	def train(self):
		
		training_set = DatasetMaper(self.x_train, self.y_train)
		test_set = DatasetMaper(self.x_test, self.y_test)
		valid_set = DatasetMaper(self.x_valid, self.y_valid)
		
		self.loader_training = DataLoader(training_set, batch_size=self.batch_size)
		self.loader_test = DataLoader(test_set)
		self.loader_valid = DataLoader(valid_set)
		
		# optimizer = optim.RMSprop(self.model.parameters(), lr=args.learning_rate)
		self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
		self.criterion = nn.BCELoss()

		for epoch in range(args.epochs):
			
			predictions = []
			
			self.model.train()
			
			for x_batch, y_batch in self.loader_training:
				
				x = x_batch.type(torch.LongTensor)
				y = y_batch.type(torch.FloatTensor)
				
				y_pred = self.model(x).squeeze()
				
				loss = self.criterion(y_pred, y)
				
				self.optimizer.zero_grad()
				
				loss.backward()
				
				self.optimizer.step()
				
				predictions += y_pred
			
			preds = torch.FloatTensor(predictions)
			valid_predictions = self.evaluation(self.loader_valid)
			
			train_accuary = self.calculate_accuray(torch.FloatTensor(self.y_train), preds)
			valid_accuracy = self.calculate_accuray(torch.FloatTensor(self.y_valid), valid_predictions)
			
			print("Epoch: %d, loss: %.2f, Train accuracy: %.2f, Validation accuracy: %.2f" % (epoch+1, loss.item(), train_accuary, valid_accuracy))


	def evaluation(self, loader):
		predictions = []
		self.model.eval()
		with torch.no_grad():
			for x_batch, y_batch in loader:
				x = x_batch.type(torch.LongTensor)
				y = y_batch.type(torch.FloatTensor)
				
				y_pred = self.model(x)
				predictions += y_pred
				
		return torch.FloatTensor(predictions)
			
	@staticmethod
	def calculate_accuray(grand_truth, predictions):
		y_pred_tag = torch.round(predictions)

		correct_pred = (y_pred_tag == grand_truth).float()
		acc = correct_pred.sum() / len(correct_pred)
		acc = torch.round(acc * 100)
        
		return acc
	
	def test(self):
		test_predictions = self.evaluation(self.loader_test)

		test_accuracy = self.calculate_accuray(torch.FloatTensor(self.y_test), test_predictions)
		print("Test accuracy: %.2f" % (test_accuracy))

		print(classification_report(self.y_test, torch.round(test_predictions)))
		confusion_matrix_df = pd.DataFrame(confusion_matrix(self.y_test, torch.round(test_predictions)))
		sns.heatmap(confusion_matrix_df, annot=True)
	
if __name__ == "__main__":
	
	args = parameter_parser()
	
	execute = Execute(args)
	execute.train()
	execute.test()