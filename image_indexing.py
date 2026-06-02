#!/usr/bin/env python3
"""Index images by visual content using pre-trained CNN features."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Sequence, Tuple


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
INDEX_COLUMNS = ["image_path", "feature"]


def load_cnn_model(model_name: str):
    """Load a pre-trained CNN model and matching preprocess function."""
    from tensorflow.keras.applications import ResNet50, VGG16
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

    name = model_name.lower()
    if name == "vgg16":
        model = VGG16(weights="imagenet", include_top=False, pooling="avg")
        return model, vgg_preprocess
    if name == "resnet":
        model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        return model, resnet_preprocess
    raise ValueError("Unsupported model. Use 'vgg16' or 'resnet'.")


def extract_feature(
    image_path: Path,
    model,
    preprocess_fn: Callable,
):
    """Extract and normalize feature vector from one image."""
    import cv2
    import numpy as np

    # OpenCV loads in BGR, convert to RGB for Keras models.
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32")
    image_batch = np.expand_dims(image, axis=0)
    image_batch = preprocess_fn(image_batch)

    # Predict feature vector and normalize it for cosine similarity.
    features = model.predict(image_batch, verbose=0).flatten()
    norm = np.linalg.norm(features)
    return features if norm == 0 else features / norm


def serialize_feature(feature) -> str:
    """Convert a numeric vector to a compact CSV-safe string."""
    return " ".join(map(str, feature.tolist()))


def deserialize_feature(feature_text: str):
    """Convert the CSV string back to a numeric vector."""
    import numpy as np

    vector = np.array(feature_text.split(), dtype=np.float32)
    if vector.size == 0:
        raise ValueError("Found an empty feature vector in index.")
    return vector


def list_images(image_dir: Path) -> List[Path]:
    """List supported images in a directory recursively."""
    return sorted(
        [
            p
            for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def build_index(image_dir: Path, output_csv: Path, model_name: str) -> None:
    """Extract features for all images in a directory and save to CSV."""
    import pandas as pd

    image_paths = list_images(image_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in: {image_dir}")

    model, preprocess_fn = load_cnn_model(model_name)
    rows = []
    for path in image_paths:
        feature = extract_feature(path, model, preprocess_fn)
        rows.append({"image_path": str(path), "feature": serialize_feature(feature)})

    dataframe = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    dataframe.to_csv(output_csv, index=False)
    print(f"Indexed {len(rows)} images into: {output_csv}")


def load_index(index_csv: Path):
    """Load indexed image paths and feature matrix from CSV."""
    import pandas as pd
    import numpy as np

    dataframe = pd.read_csv(index_csv)
    if INDEX_COLUMNS[0] not in dataframe.columns or INDEX_COLUMNS[1] not in dataframe.columns:
        raise ValueError("Index CSV must contain 'image_path' and 'feature' columns.")

    image_paths: List[str] = dataframe[INDEX_COLUMNS[0]].astype(str).tolist()
    features = np.vstack([deserialize_feature(text) for text in dataframe[INDEX_COLUMNS[1]]])
    return image_paths, features


def search_similar_images(
    query_image: Path,
    index_csv: Path,
    model_name: str,
    top_k: int,
) -> Sequence[Tuple[str, float]]:
    """Return top-k similar images with cosine similarity scores."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    model, preprocess_fn = load_cnn_model(model_name)
    query_feature = extract_feature(query_image, model, preprocess_fn)

    image_paths, indexed_features = load_index(index_csv)
    similarities = cosine_similarity([query_feature], indexed_features)[0]

    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    return [(image_paths[i], float(similarities[i])) for i in sorted_indices]


def show_results(query_image: Path, results: Sequence[Tuple[str, float]]) -> None:
    """Display query image and results with matplotlib."""
    import cv2
    import matplotlib.pyplot as plt

    total = len(results) + 1
    plt.figure(figsize=(4 * total, 4))

    # Query image panel.
    query = cv2.imread(str(query_image))
    if query is not None:
        query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
        plt.subplot(1, total, 1)
        plt.imshow(query)
        plt.title("Query")
        plt.axis("off")

    # Result panels.
    for idx, (path, score) in enumerate(results, start=2):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, total, idx)
        plt.imshow(img)
        plt.title(f"{Path(path).name}\nscore={score:.4f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Image indexing and similarity search using deep learning."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a directory of images")
    index_parser.add_argument("--image-dir", required=True, type=Path)
    index_parser.add_argument("--output-csv", required=True, type=Path)
    index_parser.add_argument(
        "--model", default="vgg16", choices=["vgg16", "resnet"], help="CNN backbone"
    )

    search_parser = subparsers.add_parser("search", help="Search similar images")
    search_parser.add_argument("--query-image", required=True, type=Path)
    search_parser.add_argument("--index-csv", required=True, type=Path)
    search_parser.add_argument(
        "--model", default="vgg16", choices=["vgg16", "resnet"], help="CNN backbone"
    )
    search_parser.add_argument("--top-k", default=5, type=int)
    search_parser.add_argument("--show", action="store_true")

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        build_index(args.image_dir, args.output_csv, args.model)
        return

    results = search_similar_images(
        query_image=args.query_image,
        index_csv=args.index_csv,
        model_name=args.model,
        top_k=args.top_k,
    )
    for rank, (path, score) in enumerate(results, start=1):
        print(f"{rank}. {path} (score={score:.6f})")
    if args.show:
        show_results(args.query_image, results)


if __name__ == "__main__":
    main()
