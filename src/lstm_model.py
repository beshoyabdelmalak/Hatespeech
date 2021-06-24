import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.ModuleList):

	def __init__(self, args):
		super().__init__()
		
		self.batch_size = args.batch_size
		self.hidden_dim = args.hidden_dim
		self.LSTM_layers = args.lstm_layers
		self.input_size = args.max_words
		self.directions = 2 if args.bidirectional else 1
		
		self.dropout = nn.Dropout(0.4)
		self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
		self.lstm1 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True, bidirectional=args.bidirectional)
		self.fc1 = nn.Linear(in_features=self.hidden_dim*self.directions, out_features=256)
		self.fc2 = nn.Linear(256, 2)
		
	def forward(self, x):
		embedded = self.embedding(x)
		lstm_out,_ = self.lstm1(embedded)
		dropout1 = self.dropout(lstm_out)
		linear_out = self.fc1(dropout1[:,-1,:])

		dropout2 = self.dropout(linear_out)
		linear2 = self.fc2(dropout2)
		# out = torch.sigmoid(linear2)
		# out = F.softmax(linear2, dim=1)

		return linear2

	def predict(self, sentence):
		sentence = self.preprocessing.sequence_to_token(sentence)
		outputs = self(torch.LongTensor(sentence))
		result = F.softmax(outputs, dim=1)
		return result