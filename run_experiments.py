from cardinality_estimation import train_GNCE
import os
import sys

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.chdir('/home/tim/cardinality_estimator/LMKG/lmkgs')
sys.path.append('home/tim/cardinality_estimator/LMKG/lmkgs')
sys.path.append('home/tim/cardinality_estimator/GCARE')

from datetime import datetime
from LMKG.lmkgs.lmkgs import run_lmkg
from GCARE.run_estimation import run_gcare
import time
import json

starttime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


query_filename = "Joined_Queries.json"

dataset = 'swdf'
query_type = 'star'

run_LMKG = True
run_GNCE = True
run_GCARE = False

# Whether to perform full inductive training
inductive = 'false' # Choices are 'false' or 'full'. false means normal training and full means evaluating without embeddings


if run_LMKG:
    print("**** Starting LMKG ****")
    starting_time_lmkg = time.time()
    # How many atoms will be trained and evaluated on
    n_atoms_lmkg = 0
    n_atoms_lmkg += run_lmkg(dataset=dataset, query_form=query_type, eval_folder=starttime, query_filename=query_filename, train=True,
             inductive=inductive)
    n_atoms_lmkg += run_lmkg(dataset=dataset, query_form=query_type, eval_folder=starttime, query_filename=query_filename, train=False,
             inductive=inductive)
    # How long training and evaluating takes per atom in ms
    total_training_time_per_atom = (time.time() - starting_time_lmkg)/n_atoms_lmkg * 1000
    print(f'Training LMKG took {total_training_time_per_atom} ms per atom')
    print(f'Trained on a total of {n_atoms_lmkg} token')

    training_timing = {'total_training_time_per_atom': total_training_time_per_atom}
    with open(f"/home/tim/Datasets/{dataset}/Results/{starttime}/LMKG/training_timing.json", 'w') as file:
        json.dump(training_timing, file, indent=4)


if run_GNCE:
    print("**** Starting GNCE ****")
    os.chdir('/home/tim/cardinality_estimator/')
    start = time.time()
    n_atoms, start_time_gnce,  end_time_gnce = train_GNCE(dataset=dataset, query_type=query_type, query_filename=query_filename, eval_folder=starttime,
               inductive=inductive)

    end = time.time()
    total_training_time_per_atom = (end - start)/n_atoms * 1000 #Note: start_time_gnce and end_time_gnce are the times for only the training loop, without data loading
    print(f'Training GNCE took {total_training_time_per_atom} ms per atom')
    print(f'Trained on a total of {n_atoms} token')
    training_timing = {'total_training_time_per_atom': total_training_time_per_atom, "n_atoms": n_atoms,
                       "total_time": (end_time_gnce - start_time_gnce) *1000 }
    with open(f"/home/tim/Datasets/{dataset}/Results/{starttime}/GNCE/training_timing.json", 'w') as file:
        json.dump(training_timing, file, indent=4)



if run_GCARE:
    print("**** Starting GCARE ****")
    os.chdir('/home/tim/cardinality_estimator/')
    run_gcare(dataset=dataset, query_type=query_type, eval_folder=starttime, query_filename=query_filename,
               inductive=inductive)
