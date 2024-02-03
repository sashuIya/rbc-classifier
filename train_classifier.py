import matplotlib

matplotlib.use("Agg")  # 'Agg' backend is suitable for saving figures to files
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from scipy.stats import mode
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from consts import (
    LABEL_UNLABELED,
    X_COLUMN_PREFIX,
    Y_COLUMN,
)
from filepath_util import (
    read_embedder_and_faiss,
    read_labeled_and_reviewed_features_for_all_images,
    read_masks_features,
    write_embedder_and_faiss,
)

DEVICE = "cuda"
BATCH_SIZE = 512


class Embedder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Embedder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            # nn.BatchNorm1d(512),  # Add batch normalization layer
            # nn.Dropout(inplace=True),
            nn.Mish(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(128),  # Add batch normalization layer
            # nn.Dropout(inplace=True),
            nn.Mish(),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


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
        for features_batch, labels_batch in data_loader:
            # features_batch = features_batch[0].to(DEVICE)
            embeddings = embedder(features_batch.to(DEVICE))
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels_batch.numpy())

    # Convert the lists to numpy arrays
    embeddings_array = np.concatenate(embeddings_list)
    labels_array = np.concatenate(labels_list)

    return embeddings_array, labels_array


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

    # Use numpy.unique() to get unique elements and their counts
    unique_elements, counts = np.unique(labeled_data_df[Y_COLUMN], return_counts=True)

    # Print the unique elements and their counts
    print("All available data:")
    for element, count in zip(unique_elements, counts):
        print(f"    Class {element}: {count} masks")

    # Create the dataset and data loader
    all_dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).int())
    all_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    optimizer = optim.Adam(embedder.parameters(), lr=0.00001)
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    num_epochs = 300

    for epoch in range(1, num_epochs + 1):
        train_embedder(
            embedder, loss_func, mining_func, DEVICE, train_loader, optimizer, epoch
        )

    test_embedder(train_dataset, val_dataset, embedder, accuracy_calculator)

    train_embeddings, train_embedding_labels = embed(embedder, train_loader)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(train_embeddings)
    # Create a scatter plot with different colors for each label
    plt.figure(figsize=(10, 8))
    for label in np.unique(train_embedding_labels):
        plt.scatter(
            embeddings_2d[train_embedding_labels == label, 0],
            embeddings_2d[train_embedding_labels == label, 1],
            label=f"Label {label_encoder.inverse_transform([label])}",
        )
    plt.title("t-SNE Visualization of Embeddings")
    plt.legend()

    # Save the plot as an image
    plt.savefig("train_embedding_visualization.png")

    ## Validate
    val_embeddings, val_embedding_labels = embed(embedder, val_loader)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(val_embeddings)
    # Create a scatter plot with different colors for each label
    plt.figure(figsize=(10, 8))
    for label in np.unique(val_embedding_labels):
        plt.scatter(
            embeddings_2d[val_embedding_labels == label, 0],
            embeddings_2d[val_embedding_labels == label, 1],
            label=f"Label {label_encoder.inverse_transform([label])}",
        )
    plt.title("t-SNE Visualization of Embeddings")
    plt.legend()

    # Save the plot as an image
    plt.savefig("val_embedding_visualization.png")

    all_embeddings, all_embedding_labels = embed(embedder, all_loader)

    all_embeddings = all_embeddings / np.linalg.norm(
        all_embeddings, axis=1, keepdims=True
    )
    index = faiss.IndexFlatIP(all_embeddings.shape[1])
    index.add(all_embeddings)

    write_embedder_and_faiss(embedder, index, all_embedding_labels, label_encoder)


def classify(df, classifier_model_filepath):
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    embedder, faiss_index, faiss_labels, label_encoder = read_embedder_and_faiss(
        classifier_model_filepath
    )
    embedder.to(DEVICE)

    df = df.loc[df[Y_COLUMN] == LABEL_UNLABELED]

    x_columns = [x for x in df.columns if x.startswith(X_COLUMN_PREFIX)]
    x = df[x_columns].to_numpy()

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
    embeddings_to_classify = np.concatenate(embeddings_list)
    embeddings_to_classify = embeddings_to_classify / np.linalg.norm(
        embeddings_to_classify, axis=1, keepdims=True
    )

    k = 5

    predicted_labels = []

    for i in range(embeddings_to_classify.shape[0]):
        query_embedding = embeddings_to_classify[i : i + 1]

        # Perform a similarity search to find nearest neighbors
        distances, indices = faiss_index.search(query_embedding.astype(np.float32), k)

        # Determine the majority label among the neighbors
        neighbor_labels = faiss_labels[indices[0]]
        mode_result = mode(neighbor_labels)
        classified_label = mode_result.mode
        predicted_labels.append(classified_label)

        print(f"Embedding {i + 1} Classified Label: {classified_label}")

    decoded_predicted_labels = label_encoder.inverse_transform(predicted_labels)
    # ! TODO: Remove this once a new classifier is trained (after 02/03/2024).
    for i, label in enumerate(decoded_predicted_labels):
        if label == "0":
            decoded_predicted_labels[i] = "red blood cell"
        if label == "1":
            decoded_predicted_labels[i] = "spheroid cell"
        if label == "2":
            decoded_predicted_labels[i] = "echinocyte"

    # Use numpy.unique() to get unique elements and their counts
    unique_elements, counts = np.unique(decoded_predicted_labels, return_counts=True)

    # Print the unique elements and their counts
    print("Predicted:")
    for element, count in zip(unique_elements, counts):
        print(f"    Class {element}: {count} masks")

    return decoded_predicted_labels


if __name__ == "__main__":
    train_pipeline()
