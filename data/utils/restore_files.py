import os
import json
import shutil
import argparse

def restore_files_from_json(json_file, source_folder, target_folder):
    with open(json_file, 'r') as f:
        folder_structure = json.load(f)

    for folder_name, files in folder_structure.items():
        folder_path = os.path.join(target_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        for file_name in files:
            source_file_path = os.path.join(source_folder, folder_name, file_name)
            target_file_path = os.path.join(folder_path, file_name)
            shutil.copyfile(source_file_path, target_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Restore files from JSON structure.')
    parser.add_argument('--json_file', type=str, help='Path to the JSON file')
    parser.add_argument('--source_folder', type=str, help='Path to the source folder')
    parser.add_argument('--target_folder', type=str, help='Path to the target folder')
    args = parser.parse_args()

    print(f'Restoring files from {args.source_folder} to {args.target_folder}...')
    restore_files_from_json(args.json_file, args.source_folder, args.target_folder)
    print('Files restored successfully!')
