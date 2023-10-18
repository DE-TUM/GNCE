import subprocess
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
import json
def remove_files_in_directory(directory_path, exclude_file='yago.txt'):
    for filename in os.listdir(directory_path):
        if filename == exclude_file:
            continue  # Skip the file to be excluded
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == '__main__':
    dataset = "yago_inductive"
    methods = ['wj', 'cset', 'jsub', 'impr']

    savetag = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    for method in methods:
        # Remove all files in ../data/dataset/data
        remove_files_in_directory(f"/home/tim/gcare/data/dataset/{dataset}", exclude_file=f'{dataset}.txt')

        starting_time = time.time()

        os.chdir("/home/tim/gcare/scripts")
        # Execute the building for the specific method
        subprocess.run(["bash", "/home/tim/gcare/scripts/run-build_call.sh", dataset, method, f'{dataset}.txt'])
        end_time = time.time() - starting_time

        # Count the number of triples in the graph at hand
        try:
            with open(f'/home/tim/Datasets/{dataset}/graph/{dataset}.nt', 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"Number of lines in graph file: {line_count}")
        except FileNotFoundError:
            print(f"Graph txt not found. Unable to count lines.")
            raise

        # There are line_count *3 atoms in the triples (with duplicates)
        total_runtime_per_atom = end_time/(line_count*3) * 1000

        Path(f"/home/tim/Datasets/{dataset}/Results/{savetag}/{method}").mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
        training_timing = {'total_training_time_per_atom': total_runtime_per_atom}
        with open(f"/home/tim/Datasets/{dataset}/Results/{savetag}/{method}/training_timing.json", 'w') as file:
            json.dump(training_timing, file, indent=4)

