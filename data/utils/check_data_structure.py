import os
import json
import argparse

data_dir = {
    "ID_ImageNet1K": ["test", "val"],
    "ID_VOC": ["test", "val"],
    "OOD_iNaturalist": ["test"],
    "OOD_Sun": ["test"],
    "OOD_Places": ["test"],
    "OOD_Texture": ["test"],
    "OOD_ImageNet22K": ["test"],
    "OOD_COCO": ["test"],
}

def generate_folder_structure_dict(target_folder):
    folder_structure = {}

    for root, dirs, files in os.walk(target_folder):
        folder_name = os.path.relpath(root, target_folder)
        image_num = 0
        if folder_name != "." and files != []:
            folder_structure[folder_name] = []
            for file in files:
                folder_structure[folder_name].append(file)
            folder_structure[folder_name].sort()
            image_num += len(folder_structure[folder_name])
    return folder_structure, image_num

def read_folder_structure_dict(json_file):
    with open(json_file, 'r') as f:
        folder_structure = json.load(f)
    empty_folders = []
    image_num = 0
    for folder_name, files in folder_structure.items():
        if files == []:
            empty_folders.append(folder_name)
        else:
            files.sort()
            image_num += len(files)
    for folder_name in empty_folders:
        del folder_structure[folder_name]
    return folder_structure, image_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='Path to datasets')
    args = parser.parse_args()
    assert os.path.exists(args.data_root), f"Path {args.data_root} does not exist."
    print("Comparing folder structure...")

    for dataset, splits in data_dir.items():
        print(f"\nComparing {dataset}...")
        for split in splits:
            json_file = f"./data/{dataset}/"+dataset.split("_")[1].lower() + f"_{split}_data.json"
            golden_dict, golden_image_num = read_folder_structure_dict(json_file)
            data_path = os.path.join(args.data_root, dataset, split)
            assert os.path.exists(data_path), f"Path {data_path} does not exist."
            generated_dict, generated_image_num = generate_folder_structure_dict(data_path)
            assert golden_dict == generated_dict, f"Split {split}: Unmatched. \nNumber of generated images: {generated_image_num}, don't match {golden_image_num}."
            print(f"Split {split}: matched! Number of images: ", golden_image_num)