import numpy as np
import pandas as pd

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
    """
    Handles batches of dataset
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Execute:
    """
    Class for execution. Initializes the preprocessing as well as the
    LSTM model
    """

    def __init__(self, args):
        self.__init_data__(args)

        self.args = args
        self.batch_size = args.batch_size

        self.model = LSTM(args)
        self.model.preprocessing = self.preprocessing

    def __init_data__(self, args):
        """
        Initialize preprocessing from raw dataset to dataset split into training and testing
        Training and test datasets are index strings that refer to tokens
        """
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
        self.loader_test = DataLoader(test_set, batch_size=self.batch_size)
        self.loader_valid = DataLoader(valid_set, batch_size=self.batch_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):

            predictions = []

            self.model.train()

            for x_batch, y_batch in self.loader_training:

                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.LongTensor)

                self.optimizer.zero_grad()

                output = self.model(x)
                loss = self.criterion(output, y)

                loss.backward()

                self.optimizer.step()

                predictions += output

            valid_predictions = self.evaluation(self.loader_valid)

            train_accuary = self.calculate_accuray(self.y_train, predictions)
            valid_accuracy = self.calculate_accuray(self.y_valid, valid_predictions)

            print(
                "Epoch: %d, loss: %.2f, Train accuracy: %.2f, Validation accuracy: %.2f"
                % (epoch + 1, loss.item(), train_accuary, valid_accuracy)
            )

    def evaluation(self, loader):
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.LongTensor)

                y_pred = self.model(x)
                predictions += y_pred

        return predictions

    @staticmethod
    def calculate_accuray_with_threshold(grand_truth, predictions):
        preds = []
        threshold = 0.5
        indexes_to_delete = []
        for i, (true, pred) in enumerate(zip(grand_truth, predictions)):
            if pred[1] > threshold:
                preds.append(1)
            elif pred[0] > threshold:
                preds.append(0)
            else:
                indexes_to_delete.append(i)

        new_truth = np.delete(grand_truth, indexes_to_delete)
        correct_pred = (
            torch.FloatTensor(preds) == torch.FloatTensor(new_truth)
        ).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return acc, preds, new_truth

    @staticmethod
    def calculate_accuray(grand_truth, predictions):
        predictions = torch.stack(predictions)
        preds = torch.argmax(predictions, dim=1)
        correct_pred = (preds == torch.FloatTensor(grand_truth)).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc * 100)
        return acc

    def test(self):
        test_predictions = self.evaluation(self.loader_test)

        test_accuracy, test_pred, true_labels = self.calculate_accuray_with_threshold(
            self.y_test, test_predictions
        )
        print("Test accuracy: %.2f" % (test_accuracy))

        print(classification_report(true_labels, test_pred))
        conf_mt = confusion_matrix(true_labels, test_pred)

        categories = ["Normal", "Hate/Offensive"]
        stats_text = ""
        if len(conf_mt) == 2:
            stats_text = create_binary_stats(conf_mt)


def create_binary_stats(confusion_matrix):
    accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
    precision = confusion_matrix[1, 1] / sum(confusion_matrix[:, 1])
    recall = confusion_matrix[1, 1] / sum(confusion_matrix[1, :])
    f1_score = 2 * precision * recall / (precision + recall)
    stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
        accuracy, precision, recall, f1_score
    )
    return stats_text


if __name__ == "__main__":

    args = parameter_parser()
    # uncomment the next three lines to retrain the model
    # execute = Execute(args)
    # execute.train()
    # execute.test()
    MODEL_PATH = "model.pth"
    # torch.save(execute.model, MODEL_PATH)
    lstm_model = torch.load(MODEL_PATH)
    lstm_model.eval()
    # read from a csv file
    # test_df = pd.read_csv("./data/real_data.csv", error_bad_lines=False)
    # test_x = test_df["text"].values
    test_x = ["Arabs are terrorists", "let's have a walk"]
    lstm_pred = lstm_model.predict(test_x)
    preds = torch.argmax(lstm_pred, dim=1)
    for sentence, pred in zip(test_x, preds):
        print(f"{sentence}: {pred}")
