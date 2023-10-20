'''
Main Script for automating manual processes of LSS training on INI Server

1. specify your dataset and query type
2. Save_all_queries to transform the corresponding queries to the LSS folder
3. transfer_queries to send the queries to the INI Server
4. run_training to start the training command

IMPORTANT: the last step spawns a tmux shell named 'mysession'. Make sure it does
not exist on the server

'''
from query_saver import save_all_queries
from server_transfer import *
from datetime import datetime


if __name__ == "__main__":
    # Define dataset and query type to use
    dataset = "wikidata"
    query_type = "path"

    starttime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    initiate_training = True
    pull_training_results = False

    if initiate_training:
        # Transform the queries to G-Care format and save locally in LSS directory
        save_all_queries(dataset, query_type, ["Joined_Queries.json"])

        # Transfer Queries to remote server for training
        transfer_queries(dataset)

        # Start training on remote server
        run_training(dataset)

    if pull_training_results:
        # Pull results from remote server
        gather_training_files(dataset, f"{starttime}_{dataset}_{query_type}")


