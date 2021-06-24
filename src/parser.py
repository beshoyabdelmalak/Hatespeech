import argparse

def parameter_parser():

	parser = argparse.ArgumentParser(description = "Hatespeech Classification")

	parser.add_argument("--epochs",
						dest = "epochs",
						type = int,
						default = 5,
						help = "Number of gradient descent iterations. Default is 100.")

	parser.add_argument("--learning_rate",
						dest = "learning_rate",
						type = float,
						default = 0.001,
						help = "Gradient descent learning rate. Default is 0.01.")

	parser.add_argument("--hidden_dim",
						dest = "hidden_dim",
						type = int,
						default = 128,
						help = "Number of neurons by hidden layer. Default is 128.")
						
	parser.add_argument("--lstm_layers",
						dest = "lstm_layers",
						type = int,
						default = 1,
						help = "Number of LSTM layers")
					
	parser.add_argument("--batch_size",
						dest = "batch_size",
						type = int,
						default = 128,
						help = "Batch size")
						
	parser.add_argument("--max_len",
						dest = "max_len",
						type = int,
						default = 200,
						help = "Maximum sequence length per tweet")
						
	parser.add_argument("--max_words",
						dest = "max_words",
						type = float,
						default = 1000,
						help = "Maximum number of words in the dictionary")
	
	parser.add_argument("--bidirectional",
						dest = "bidirectional",
						type = bool,
						default = True,
						help = "Whether if lstm bidirectional or not")
				 
	return parser.parse_args()
