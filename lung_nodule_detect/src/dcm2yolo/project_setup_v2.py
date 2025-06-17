import os
from pathlib import Path


def setup_project_structure_v2(output_dir_root):
    """
    Creates the directory structure for the YOLO dataset.
    """
    base_path = Path(output_dir_root)
    base_path.mkdir(parents=True, exist_ok=True)

    dataset_name = "LIDC_YOLO_Processed_Dataset"
    dataset_root_path = base_path / dataset_name

    paths_to_create = [
        dataset_root_path / "images" / "train",
        dataset_root_path / "images" / "val",
        dataset_root_path / "images" / "test",
        dataset_root_path / "labels" / "train",
        dataset_root_path / "labels" / "val",
        dataset_root_path / "labels" / "test",
    ]

    for p in paths_to_create:
        p.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {p}")

    yaml_content = f"""
# LIDC-IDRI Lung Nodule Detection Dataset (Processed V2)
path: {dataset_root_path.resolve()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test # test images (optional)

# Classes
names:
  0: nodule

nc: 1  # number of classes
"""
    yaml_path = dataset_root_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    print(f"Created dataset YAML: {yaml_path}")

    print(f"✅ Project structure created at: {dataset_root_path}")
    return dataset_root_path

# Example usage (typically called from the main script)
# if __name__ == "__main__":
#     output_directory = r"D:\LIDC_YOLO_Output_V2"
#     setup_project_structure_v2(output_directory)