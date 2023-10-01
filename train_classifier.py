import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from filepath_util import (
    read_masks_features,
    read_embedder_and_classifier,
    write_embedder_and_classifier,
    read_labeled_and_reviewed_features_for_all_images,
)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


from consts import (
    X_COLUMN_PREFIX,
    Y_COLUMN,
    LABEL_UNLABELED,
)

DEVICE = "cuda"
BATCH_SIZE = 512


class Embedder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Embedder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),  # Add batch normalization layer
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),  # Add batch normalization layer
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_sample = self.x_data[idx]
        y_sample = self.y_data[idx]

        return x_sample, y_sample


def train_embedder(
    model, loss_func, mining_func, device, train_loader, optimizer, epoch
):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
    if epoch % 20 == 0:
        print(
            "Epoch {}  Loss = {}, Number of mined triplets = {}".format(
                epoch, loss, mining_func.num_triplets
            )
        )


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester(dataloader_num_workers=1)
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test_embedder(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))


def embed(embedder, data_loader):
    # Set the model to evaluation mode
    embedder.eval()

    # Create empty lists to store embeddings and corresponding labels
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for features_batch in data_loader:
            features_batch = features_batch[0].to(DEVICE)
            embeddings = embedder(features_batch)
            embeddings_list.append(embeddings.cpu().numpy())

    # Convert the lists to numpy arrays
    embeddings_array = np.concatenate(embeddings_list)

    return embeddings_array


def get_labels_from_data_loader(data_loader):
    # Create empty lists to store embeddings and corresponding labels
    labels_list = []

    with torch.no_grad():
        for _, labels_batch in data_loader:
            labels_list.append(labels_batch.numpy())

    return np.concatenate(labels_list)


def train_pipeline():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    labeled_data_df = read_labeled_and_reviewed_features_for_all_images(check_data=True)
    assert (
        labeled_data_df.loc[labeled_data_df[Y_COLUMN] == LABEL_UNLABELED].shape[0] == 0
    ), "Should not contain unlabeled data"

    x_columns = [x for x in labeled_data_df.columns if x.startswith(X_COLUMN_PREFIX)]
    x_size = len(x_columns)

    x = labeled_data_df[x_columns].to_numpy()
    y_labels = labeled_data_df[Y_COLUMN].to_numpy()
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    y_size = len(label_encoder.classes_)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

    # Create the dataset and data loader
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(), torch.from_numpy(y_train).int()
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test).int(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    # Create the model instance
    embedder = Embedder(x_size, y_size).to(DEVICE)
    optimizer = optim.Adam(embedder.parameters(), lr=0.00005)
    loss_func = losses.TripletMarginLoss(margin=1.0)
    mining_func = miners.TripletMarginMiner(margin=1.0, type_of_triplets="semihard")
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    num_epochs = 1200

    for epoch in range(1, num_epochs + 1):
        train_embedder(
            embedder, loss_func, mining_func, DEVICE, train_loader, optimizer, epoch
        )

    # test_embedder(train_dataset, val_dataset, embedder, accuracy_calculator)

    train_embeddings = embed(embedder, train_loader)
    train_embedding_labels = get_labels_from_data_loader(train_loader)

    # Train a linear SVM classifier
    # classifier = SVC(kernel="linear", C=1.0, random_state=42)
    classifier = KNeighborsClassifier(n_neighbors=8)
    classifier.fit(train_embeddings, train_embedding_labels)

    ## Validate
    val_embeddings = embed(embedder, val_loader)
    val_embedding_labels = get_labels_from_data_loader(val_loader)
    # Use the SVM classifier to predict labels for the validation set
    val_predicted_labels = classifier.predict(val_embeddings)
    train_predicted_labels = classifier.predict(train_embeddings)

    # Calculate accuracy on the validation set
    train_accuracy = accuracy_score(train_embedding_labels, train_predicted_labels)
    val_accuracy = accuracy_score(val_embedding_labels, val_predicted_labels)
    print(
        f"Train accuracy: {train_accuracy:.4f}\nValidation Accuracy: {val_accuracy:.4f}"
    )

    write_embedder_and_classifier(embedder, classifier, label_encoder)


def classify(df, classifier_model_filepath):
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    embedder, classifier, label_encoder = read_embedder_and_classifier(
        classifier_model_filepath
    )
    embedder.to(DEVICE)

    df = df.loc[df[Y_COLUMN] == LABEL_UNLABELED]

    x_columns = [x for x in df.columns if x.startswith(X_COLUMN_PREFIX)]
    x = df[x_columns].to_numpy()

    print("Classifying {} masks".format(x.shape[0]))

    # Create the dataset and data loader
    dataset = TensorDataset(torch.from_numpy(x).float())
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Set the model to evaluation mode
    embedder.eval()

    # Create empty lists to store embeddings and corresponding labels
    embeddings_list = []

    with torch.no_grad():
        for features_batch in data_loader:
            features_batch = features_batch[0].to(DEVICE)
            embeddings = embedder(features_batch)
            embeddings_list.append(embeddings.cpu().numpy())

    # Convert the lists to numpy arrays
    embeddings_array = np.concatenate(embeddings_list)

    predicted_labels = classifier.predict(embeddings_array)

    decoded_predicted_labels = label_encoder.inverse_transform(predicted_labels)

    # Use numpy.unique() to get unique elements and their counts
    unique_elements, counts = np.unique(decoded_predicted_labels, return_counts=True)

    # Print the unique elements and their counts
    print("Predicted:")
    for element, count in zip(unique_elements, counts):
        print(f"    Class {element}: {count} masks")

    return decoded_predicted_labels


if __name__ == "__main__":
    train_pipeline()
