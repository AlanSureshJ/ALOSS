import fiftyone as fo
from fiftyone import types

# Load the training dataset
dataset = fo.Dataset.from_dir(
    dataset_dir="C:/Users/alans/Downloads",
    dataset_type=types.BDDDataset,
    data_path="C:/Users/alans/Downloads/images/100k/train",
    labels_path="C:/Users/alans/Downloads/labels/bdd100k_labels_images_train.json",
    name="bdd100k-train"
)

dataset.export(
    export_dir="C:/Users/alans/Downloads/yolo_export",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="detections",
    split="train"
)
# Optional: Launch FiftyOne app
session = fo.launch_app(dataset, port=5152)
session.wait()

# Export to YOLOv5 format

