import subprocess
from pathlib import Path

REMOTE_ADDRESS = 'XXX'

def transfer_queries(dataset):
    print("Transfering queries to server...")
    command = f"rsync -avz /home/tim/LSS/data/queryset_homo/{dataset}/paths_1 schwatkm@{REMOTE_ADDRESS}:/work/schwatkm/lss/data/queryset_homo/{dataset}"
    # Execute the command
    subprocess.run(command, shell=True)

    command = f"rsync -avz /home/tim/LSS/data/true_homo/{dataset}/paths_1 schwatkm@{REMOTE_ADDRESS}:/work/schwatkm/lss/data/true_homo/{dataset}"

    subprocess.run(command, shell=True)
    print("Transfering queries finished")


def run_training(dataset_name):
    print("Transfering Training Command to Server...")
    command = (f'ssh schwatkm@{REMOTE_ADDRESS} "export PATH=/home/schwatkm/miniconda3/bin:$PATH && source activate'
               f' gpcard && cd /work/schwatkm/lss && tmux new-session -d -s mysession && tmux send-keys -t mysession \\"python active_train.py --dataset {dataset_name} --embed_type prone --mode train --no-cuda\\" C-m"')


    # Execute the SSH command
    subprocess.run(command, shell=True)
    print("Training Transfer finished")

def gather_training_files(dataset, starttime):
    print("Pulling Training Files from Server")

    # Your source and destination paths
    src_folder = "/work/schwatkm/lss"
    dst_folder = f"/home/tim/Datasets/{dataset}/Results/{starttime}"
    dst_models_folder = f"/home/tim/Datasets/{dataset}/Results/{starttime}/models"

    # Create destination folders if they don't exist
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    Path(dst_models_folder).mkdir(parents=True, exist_ok=True)

    # Files to copy
    files_to_copy = ["summary.txt", "pred_times_total.npy", "pred_times_lss.npy", "sizes_lss.npy", "gts_lss.npy", "preds_lss.npy"]

    # Copy each file
    for file in files_to_copy:
        subprocess.run([
            "scp",
            f"schwatkm@{REMOTE_ADDRESS}:{src_folder}/{file}",
            f"{dst_folder}/{file}"
        ])

    # To get the newest file in /work/schwatkm/lss/models/{dataset}
    # Note: This assumes the newest file is the one with the latest modification time.
    command_to_get_newest_file = (
        f"ssh schwatkm@{REMOTE_ADDRESS} 'ls -t {src_folder}/models/{dataset} | head -n 1'"
    )
    result = subprocess.run(command_to_get_newest_file, shell=True, capture_output=True, text=True)
    newest_file = result.stdout.strip()

    # Copy the newest file
    if newest_file:
        subprocess.run([
            "scp",
            f"schwatkm@{REMOTE_ADDRESS}:{src_folder}/models/{dataset}/{newest_file}",
            f"{dst_models_folder}/{newest_file}"
        ])
    else:
        print("No files found in the models directory.")



if __name__ == "__main__":
    gather_training_files("yago", "TEST")
    #transfer_queries('yago')

    #run_training("yago")
