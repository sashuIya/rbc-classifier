import io
import os
from dataclasses import dataclass

import matplotlib
from PIL import Image
from torchvision.transforms import Resize

matplotlib.use("Agg")  # 'Agg' backend is suitable for saving figures to files
from typing import List, Tuple

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
    decoded_labels: List[str]  # these are actual names (e.g., 'echinocyte')
    confidence_scores: List[float]


def _compute_knn_confidence_with_distances(neighbor_labels, similarities):
    # Calculate weights based on cosine similarities (they are in [-1, 1])
    # range.
    weights = [(similarity + 1.0) / 2.0 for similarity in similarities]

    # Compute weighted votes for each class
    unique_labels = np.unique(neighbor_labels)
    weighted_votes = {label: 0.0 for label in unique_labels}
    for label, weight in zip(neighbor_labels, weights):
        weighted_votes[label] += weight

    # Determine the mode (most frequent label)
    mode_label = max(weighted_votes, key=weighted_votes.get)

    # Calculate confidence score for the mode label
    confidence_score = weighted_votes[mode_label] / sum(weighted_votes.values())

    return mode_label, confidence_score


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

        # Perform a similarity search to find nearest neighbors in cosine space.
        similarities, indices = faiss_index.search(
            query_embedding.astype(np.float32), options.k
        )

        encoded_label, confidence_score = _compute_knn_confidence_with_distances(
            faiss_labels[indices[0]], similarities[0]
        )
        classified_label = label_encoder.inverse_transform([encoded_label])[0]

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
    embeddings_array = embeddings_array / np.linalg.norm(
        embeddings_array, axis=1, keepdims=True
    )
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

        global_step = batch_idx + epoch * len(train_loader)
        tensorboard_writer.add_scalar(
            "Loss/Train", loss.item(), global_step=global_step
        )
        tensorboard_writer.add_scalar(
            "Mined triplets", mining_func.num_triplets, global_step=global_step
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


def _calculate_target_size(images_list: List[np.ndarray]) -> Tuple[int, int]:
    """
    Calculates the target size for resizing images based on the maximum height and width.
    """
    heights = [img.shape[0] for img in images_list] + [0]
    widths = [img.shape[1] for img in images_list] + [0]
    max_height = max(heights)
    max_width = max(widths)
    max_value = max(max_height, max_width)

    # Tensorboard supports only square images.
    return max_value, max_value


def _convert_images_to_tensor(images_list: List[np.ndarray]) -> torch.Tensor:
    """
    Converts a list of NumPy arrays representing images into a PyTorch tensor.
    All images are resized to the maximum height and width before stacking.

    Parameters:
    - images_list: List of NumPy arrays, where each array represents an image.

    Returns:
    - A PyTorch tensor with shape (n, c, h, w), where n is the number of images, c is the number of channels, h is the height, and w is the width.
    """
    # Calculate the target size based on the maximum height and width
    target_size = _calculate_target_size(images_list)

    # Initialize a list to hold the resized images
    resized_images = []

    # Iterate over each image in the list
    for image in images_list:
        # Convert the NumPy array to a PIL Image
        pil_image = Image.fromarray(image)

        # Resize the image
        resized_image = Resize(target_size)(pil_image)

        # Convert the resized PIL Image back to a NumPy array
        resized_array = np.array(resized_image)

        # Append the resized array to the list
        resized_images.append(resized_array)

    # Convert the list of NumPy arrays to a list of PyTorch tensors
    tensors_list = [torch.from_numpy(image) for image in resized_images]

    # Stack the list of tensors along a new dimension to form a single tensor
    images_tensor = torch.stack(tensors_list)

    # Transpose the tensor to move the channel dimension to the second position
    images_tensor = images_tensor.permute(0, 3, 1, 2)

    return images_tensor


def _plot_clusters(
    tensorboard_writer: SummaryWriter,
    data_loader: DataLoader,
    crops: List[np.ndarray],
    output_filename: str,
    embedder: Embedder,
    label_encoder: preprocessing.LabelEncoder,
    log_images_to_tensorboard: bool = False,
):
    """Computes the embeddings for the given data_loader, plots and saves 2d projections."""
    embeddings, embedding_labels = _embed(embedder, data_loader, read_labels=True)
    if log_images_to_tensorboard:
        images_tensor = _convert_images_to_tensor(crops)
        tensorboard_writer.add_embedding(
            embeddings,
            label_img=images_tensor.float() / 255.0,
            metadata=label_encoder.inverse_transform(embedding_labels),
            tag=output_filename,
        )

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


def _convert_confusion_matrix_to_img(cm, labels, title):
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


def _create_confusion_matrix(
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


@timeit
def log_confusion_matrices_to_tensorboard(
    tensorboard_writer,
    train_loader,
    val_loader,
    embedder,
    faiss_index,
    faiss_labels,
    label_encoder,
    knn_options,
    step,
):
    train_confusion_matrix = _create_confusion_matrix(
        train_loader, embedder, faiss_index, faiss_labels, label_encoder, knn_options
    )
    tensorboard_writer.add_image(
        "Train Confusion Matrix",
        _convert_confusion_matrix_to_img(
            train_confusion_matrix, label_encoder.classes_, "Train Confusion Matrix"
        ),
        step,
        dataformats="HWC",
    )

    val_confusion_matrix = _create_confusion_matrix(
        val_loader, embedder, faiss_index, faiss_labels, label_encoder, knn_options
    )
    tensorboard_writer.add_image(
        "Validation Confusion Matrix",
        _convert_confusion_matrix_to_img(
            val_confusion_matrix, label_encoder.classes_, "Validation Confusion Matrix"
        ),
        step,
        dataformats="HWC",
    )


@timeit
def log_train_and_val_clusters(
    tensorboard_writer: SummaryWriter,
    train_loader: DataLoader,
    train_crops: List[np.ndarray],
    val_loader: DataLoader,
    val_crops: List[np.ndarray],
    embedder: nn.Module,
    label_encoder: preprocessing.LabelEncoder,
):
    # Plot and save train and val embeddings (as 2d projections).
    try:
        _plot_clusters(
            tensorboard_writer,
            train_loader,
            train_crops,
            "train_embedding_visualization.png",
            embedder,
            label_encoder,
            log_images_to_tensorboard=False,
        )
    except Exception as e:
        print(f"An unexpected error occurred when plotting train clusters: {e}")

    try:
        _plot_clusters(
            tensorboard_writer,
            val_loader,
            val_crops,
            "val_embedding_visualization.png",
            embedder,
            label_encoder,
            log_images_to_tensorboard=False,
        )
    except Exception as e:
        print(f"An unexpected error occurred when plotting validation clusters: {e}")


@timeit
def train_pipeline(dir: str = None, classifier_model_filepath: str = None):
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    embedder_model_info = EmbedderModelInfo()
    tensorboard_writer = SummaryWriter(
        log_dir=os.path.join(CLASSIFIER_TENSORBOARD_DIR, embedder_model_info.name)
    )

    labeled_data_df, crops = read_labeled_and_reviewed_features_for_all_images(
        dir=dir, with_crops=True, check_data=True
    )
    if labeled_data_df.empty:
        print("No labeled data found. Exiting.")
        return

    assert labeled_data_df.loc[
        labeled_data_df[Y_COLUMN] == LABEL_UNLABELED
    ].empty, "Should not contain unlabeled data"

    x_columns = [x for x in labeled_data_df.columns if x.startswith(X_COLUMN_PREFIX)]

    x = labeled_data_df[x_columns].to_numpy()
    y_labels = labeled_data_df[Y_COLUMN].to_numpy()

    # Create or load model instance and label encoder.
    is_in_fine_tune_mode = classifier_model_filepath is not None
    if is_in_fine_tune_mode:
        embedder, _, _, label_encoder = read_embedder_and_faiss(
            classifier_model_filepath
        )
    else:
        label_encoder = preprocessing.LabelEncoder()
        embedder = Embedder(len(x_columns), len(np.unique(y_labels)))
    embedder.to(DEVICE)

    y = label_encoder.fit_transform(y_labels)
    train_crops = None
    val_crops = None
    if len(crops) == len(x):
        x_train, x_test, y_train, y_test, train_crops, val_crops = train_test_split(
            x, y, crops, random_state=10
        )
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

    # Use numpy.unique() to get unique elements and their counts
    unique_elements, counts = np.unique(labeled_data_df[Y_COLUMN], return_counts=True)

    # Print the unique elements and their counts
    print("All available data:")
    for element, count in zip(unique_elements, counts):
        print(f"    Class {element}: {count} masks")

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

    optimizer = optim.Adam(embedder.parameters(), lr=0.00001)
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    num_epochs = 100 if is_in_fine_tune_mode else 300
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

    faiss_index, faiss_labels = _create_faiss_index(train_loader, embedder)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    log_train_and_val_clusters(
        tensorboard_writer,
        train_loader,
        train_crops,
        val_loader,
        val_crops,
        embedder,
        label_encoder,
    )
    log_confusion_matrices_to_tensorboard(
        tensorboard_writer,
        train_loader,
        val_loader,
        embedder,
        faiss_index,
        faiss_labels,
        label_encoder,
        knn_options=KnnClassifyOptions(),
        step=num_epochs * len(train_loader),
    )
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
