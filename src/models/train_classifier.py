import io
import os
from dataclasses import dataclass

import matplotlib
from PIL import Image

matplotlib.use("Agg")  # 'Agg' backend is suitable for saving figures to files
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from scipy.stats import mode
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.common.consts import (
    CLASSIFIER_TENSORBOARD_DIR,
    LABEL_UNLABELED,
    X_COLUMN_PREFIX,
    Y_COLUMN,
)
from src.common.filepath_util import (
    EmbedderModelInfo,
    read_embedder_and_faiss,
    read_labeled_and_reviewed_features_for_all_images,
    write_embedder_and_faiss,
)
from src.utils.timing import timeit

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


@dataclass
class KnnClassifyOptions:
    k: int = 10
    min_votes: int = 3


@dataclass
class ClassifyResults:
    decoded_labels: list[str]  # these are actual names (e.g., 'echinocyte')
    confidence_scores: list[float]


def _classify_embeddings(
    embeddings, faiss_index, faiss_labels, label_encoder, options: KnnClassifyOptions
) -> ClassifyResults:
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # This is for actual names ('echinocyte', 'intermediate mainly biconcave RBC',
    # etc).
    decoded_predicted_labels = []
    confidence_scores = []

    for i in range(embeddings.shape[0]):
        query_embedding = embeddings[i : i + 1]

        # Perform a similarity search to find nearest neighbors.
        distances, indices = faiss_index.search(
            query_embedding.astype(np.float32), options.k
        )

        # Determine the majority label among the neighbors.
        neighbor_labels = faiss_labels[indices[0]]
        mode_result = mode(neighbor_labels)

        confidence_score = compute_confidence_score(
            distances, neighbor_labels, mode_result
        )
        classified_label = LABEL_UNLABELED
        if mode_result.count >= options.min_votes:
            classified_label = label_encoder.inverse_transform([mode_result.mode])[0]
        decoded_predicted_labels.append(classified_label)
        confidence_scores.append(confidence_score)

    return ClassifyResults(
        decoded_labels=decoded_predicted_labels, confidence_scores=confidence_scores
    )


def _embed(embedder: nn.Module, data_loader: DataLoader, read_labels: bool):
    # Set the model to evaluation mode
    embedder.eval()

    # Create empty lists to store embeddings and corresponding labels
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in data_loader:
            features_batch = batch[0]
            embeddings = embedder(features_batch.to(DEVICE))
            embeddings_list.append(embeddings.cpu().numpy())
            if read_labels:
                labels_batch = batch[1]
                labels_list.append(labels_batch.numpy())

    # Convert the lists to numpy arrays
    embeddings_array = np.concatenate(embeddings_list)

    if read_labels:
        labels_array = np.concatenate(labels_list)
        return embeddings_array, labels_array

    return embeddings_array, None


def _create_faiss_index(data_loader, embedder):
    embeddings, labels = _embed(embedder, data_loader, read_labels=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, labels


def train_embedder(
    model,
    loss_func,
    mining_func,
    device,
    train_loader,
    optimizer,
    epoch,
    tensorboard_writer,
):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()

        tensorboard_writer.add_scalar(
            "Loss/Train",
            loss.item(),
            global_step=batch_idx + epoch * len(train_loader),
        )

        optimizer.step()

    if epoch % 20 == 0:
        print(
            "Epoch {}  Loss = {}, Number of mined triplets = {}".format(
                epoch, loss, mining_func.num_triplets
            )
        )


def validate_embedder(
    model, data_loader, loss_func, device, tensorboard_writer, global_step
):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No need to track gradients during validation
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = loss_func(outputs, targets)
            total_loss += loss.item()

    avg_validation_loss = total_loss / len(data_loader.dataset)
    if (global_step + 1) % 20 == 0:
        print(f"Loss/Validation: {avg_validation_loss:.4f}")

    # Log the validation loss to TensorBoard
    tensorboard_writer.add_scalar(
        "Loss/Validation", avg_validation_loss, global_step=global_step
    )


def _get_all_embeddings(dataset, model):
    """Convenient function from pytorch-metric-learning."""
    tester = testers.BaseTester(dataloader_num_workers=1)
    return tester.get_all_embeddings(dataset, model)


def test_embedder(train_set, test_set, model, accuracy_calculator) -> float:
    """Computes and returns accuracy using AccuracyCalculator from pytorch-metric-learning."""
    train_embeddings, train_labels = _get_all_embeddings(train_set, model)
    test_embeddings, test_labels = _get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)

    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )

    accuracy = accuracies["precision_at_1"]
    print("Test set accuracy (Precision@1) = {}".format(accuracy))

    return accuracy


def get_labels_from_data_loader(data_loader):
    # Create empty lists to store embeddings and corresponding labels
    labels_list = []

    with torch.no_grad():
        for _, labels_batch in data_loader:
            labels_list.append(labels_batch.numpy())

    return np.concatenate(labels_list)


def plot_clusters(
    data_loader: DataLoader,
    output_filename: str,
    embedder: Embedder,
    label_encoder: preprocessing.LabelEncoder,
):
    """Computes the embeddings for the given data_loader, plots and saves 2d projections."""
    embeddings, embedding_labels = _embed(embedder, data_loader, read_labels=True)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    # Create a scatter plot with different colors for each label
    plt.figure(figsize=(10, 8))
    for label in np.unique(embedding_labels):
        plt.scatter(
            embeddings_2d[embedding_labels == label, 0],
            embeddings_2d[embedding_labels == label, 1],
            label=f"Label {label_encoder.inverse_transform([label])}",
        )
    plt.title("t-SNE Visualization of Embeddings")
    plt.legend()

    # Save the plot as an image
    plt.savefig(output_filename)


def convert_confusion_matrix_to_img(cm, labels, title):
    """Converts a confusion matrix to an image."""
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )

    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Reduce label font size for better fitting
    ax.tick_params(axis="both", which="major", labelsize=6)

    # Rotate labels for better visibility
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)

    # Convert the Matplotlib figure to a NumPy array
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    img_pil = Image.open(buf)

    # Remove the alpha channel if present
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    img_array = np.array(img_pil)

    return img_array


def create_confusion_matrix(
    data_loader: DataLoader,
    embedder: nn.Module,
    faiss_index,
    faiss_labels,
    label_encoder,
    knn_options: KnnClassifyOptions,
):
    embeddings, labels = _embed(embedder, data_loader, read_labels=True)
    true_decoded_labels = label_encoder.inverse_transform(labels)
    classify_results = _classify_embeddings(
        embeddings, faiss_index, faiss_labels, label_encoder, knn_options
    )

    return confusion_matrix(
        true_decoded_labels,
        classify_results.decoded_labels,
        labels=label_encoder.classes_,
    )


def log_confusion_matrices_to_tensorboard(
    tensorboard_writer,
    train_loader,
    val_loader,
    embedder,
    label_encoder,
    knn_options,
    step,
):
    faiss_index, faiss_labels = _create_faiss_index(train_loader, embedder)

    train_confusion_matrix = create_confusion_matrix(
        train_loader, embedder, faiss_index, faiss_labels, label_encoder, knn_options
    )
    tensorboard_writer.add_image(
        "Train Confusion Matrix",
        convert_confusion_matrix_to_img(
            train_confusion_matrix, label_encoder.classes_, "Train Confusion Matrix"
        ),
        step,
        dataformats="HWC",
    )

    val_confusion_matrix = create_confusion_matrix(
        val_loader, embedder, faiss_index, faiss_labels, label_encoder, knn_options
    )
    tensorboard_writer.add_image(
        "Validation Confusion Matrix",
        convert_confusion_matrix_to_img(
            val_confusion_matrix, label_encoder.classes_, "Validation Confusion Matrix"
        ),
        step,
        dataformats="HWC",
    )


def plot_train_and_val_clusters(train_loader, val_loader, embedder, label_encoder):
    # Plot and save train and val embeddings (as 2d projections).
    plot_clusters(
        train_loader,
        "train_embedding_visualization.png",
        embedder,
        label_encoder,
    )

    try:
        plot_clusters(
            val_loader,
            "val_embedding_visualization.png",
            embedder,
            label_encoder,
        )
    except Exception as e:
        print(f"An unexpected error occurred when plotting validation clusters: {e}")


def train_pipeline(dir: str = None):
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    embedder_model_info = EmbedderModelInfo()
    tensorboard_writer = SummaryWriter(
        log_dir=os.path.join(CLASSIFIER_TENSORBOARD_DIR, embedder_model_info.name)
    )
    labeled_data_df = read_labeled_and_reviewed_features_for_all_images(
        dir, check_data=True
    )
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
    skip_evaluation = 5

    for epoch in range(1, num_epochs + 1):
        train_embedder(
            embedder,
            loss_func,
            mining_func,
            DEVICE,
            train_loader,
            optimizer,
            epoch,
            tensorboard_writer,
        )

        if (epoch + 1) % skip_evaluation == 0:
            global_step = epoch * len(train_loader)
            validate_embedder(
                model=embedder,
                data_loader=val_loader,
                loss_func=loss_func,
                device=DEVICE,
                tensorboard_writer=tensorboard_writer,
                global_step=global_step,
            )

    validation_accuracy = test_embedder(
        train_dataset, val_dataset, embedder, accuracy_calculator
    )

    plot_train_and_val_clusters(train_loader, val_dataset, embedder, label_encoder)
    log_confusion_matrices_to_tensorboard(
        tensorboard_writer,
        train_loader,
        val_loader,
        embedder,
        label_encoder,
        knn_options=KnnClassifyOptions(),
        step=num_epochs * len(train_loader),
    )

    faiss_index, faiss_labels = _create_faiss_index(all_loader, embedder)
    write_embedder_and_faiss(
        embedder,
        embedder_model_info,
        faiss_index,
        faiss_labels,
        label_encoder,
        labeled_data_df=labeled_data_df,
        validation_accuracy=round(validation_accuracy, 2),
        epochs=num_epochs,
    )


def compute_confidence_score(distances, neighbor_labels, mode_result):
    """
    Compute the confidence score for the classification decision based on the distances
    and labels of the nearest neighbors.

    Parameters:
    - distances: Array-like object containing the distances of the nearest neighbors.
    - neighbor_labels: Array-like object containing the labels of the nearest neighbors.
    - mode_result: Result from scipy.stats.mode, containing the mode (most common label)
                   and its count among the nearest neighbors.

    Returns:
    - confidence_score: Float representing the confidence score for the classification decision.
    """
    # Normalize distances to sum to 1 for easier interpretation
    normalized_distances = distances / np.sum(distances)

    # Find indices of neighbors with the majority label
    majority_label_indices = np.where(neighbor_labels == mode_result.mode)

    # Extract corresponding normalized distances
    relevant_normalized_distances = normalized_distances[0][majority_label_indices]

    # Compute the average of these distances to get a single confidence score
    average_confidence_score = 1 - np.mean(relevant_normalized_distances)

    return average_confidence_score


@timeit
def classify(
    df: pd.DataFrame,
    classifier_model_filepath: str,
    options: KnnClassifyOptions = KnnClassifyOptions(),
) -> ClassifyResults:
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

    embeddings, _ = _embed(embedder, data_loader, read_labels=False)
    classify_results = _classify_embeddings(
        embeddings, faiss_index, faiss_labels, label_encoder, options
    )

    # Print the predicted classes and their counts.
    class_names, counts = np.unique(classify_results.decoded_labels, return_counts=True)
    print("Predicted:")
    print("{:<50} {:<8}".format("Class", "Count"))
    for class_name, count in zip(class_names, counts):
        print(f"{class_name:<50} {count:<8}")

    return classify_results


if __name__ == "__main__":
    train_pipeline()
