import sys
sys.path.append('/home/tim/cardinality_estimator_publication/LMKG/lmkgs')

import argparse
import os, time, numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from data_processor_release import read_star_graph_pattern, read_chain_graph_pattern, read_combined, read_combined_all_sizes_star_or_chain
from estimates_file import check_network_estimates
from tensorflow.keras import backend as K
import complex_reader
import store_statistics
from pathlib import Path
from tensorflow.keras.callbacks import LambdaCallback
from tqdm import tqdm
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


with tf.device("GPU:0"):
    def inverse_transform_minmax(y):
        """ Returns the inverse transform of the number """
        y = denormalize_MINMAX(y)
        y = K.exp(y)
        return y

    def q_loss(y_true, y_pred):
        """ Calculation of q_loss with the original values """
        y_true = inverse_transform_minmax(y_true)
        y_pred = inverse_transform_minmax(y_pred)
        return K.maximum(y_true, y_pred) / K.minimum(y_true, y_pred)

    def write_MIN_MAX(file_name, MIN, MAX):
        with open("final_datasets/minmax/" + file_name, "w") as f:
            f.write(str(MIN)+" "+str(MAX))

    def normalize(y, min, max):
        """ Normalization used for the cardinality """
        y = (y - min) / (max - min)
        return y

    def denormalize(y, min, max):
        """ Denormalization used for the cardinality """
        y = y * (max - min) + min
        return y


    def create_flatten_mlp(shape, include_layer = False, regress = False, additional_layers = [256, 256]):
        """
        Creates Flattening and possible adds an MLP Layer for X, A, E separately
        :param shape: [input_shape] of the input that needs to be passed by the layers
        :param include_layer: boolean, indicates whether we are adding MLP for the specific input
        :param regress: boolean, parameter used mainly for testing
        :param additional_layers: array of neurons for the additional MLP
        :return: model
        """

        total = 1
        for s in shape:
            total *= s
        model = Sequential()
        model.add(Flatten(input_shape=shape))

        if include_layer:
            for i in range(1, len(additional_layers)):
                model.add(Dense(additional_layers[i], activation="relu"))
            # model.add(Dense(128, activation="relu"))
        # check to see if the regression node should be added
        if regress:
            model.add(Dense(1, activation="linear"))
        # return our model
        return model


    def normalize_MINMAX(y):
        """ Normalization with MIN and MAX value """
        if 'MIN' not in globals():
            global MIN, MAX
            MIN = min(y)
            MAX = max(y)
        y = (y - MIN) / (MAX - MIN)
        return y

    def denormalize_MINMAX(y):
        """ Denormalization with MIN and MAX value """
        y = (y * (MAX - MIN)) + MIN
        return y

    def read_MIN_MAX(file_name):
        """ Reading of MIN and MAX value needed for prediction time since we require the same scaling as during training"""
        with open("final_datasets/minmax/" + file_name, "r") as f:
            line = f.readline()
            min_y = int(line.split(" ")[0])
            max_y = int(line.split(" ")[1])
            return min_y, max_y

    def create_model(matrix_mode, d, b, n, e, layers = [1024, 1024]):
        """
        Creates Flattening and possible adds an MLP Layer for X, A, E separately
        :param matrix_mode: 0,1 being two matrices 2 and 3 being with 3 matrices as reported in paper, one-hot and binary encoding, 4 works only for a specific query
        :param d: int, the number of distinct nodes (subjects + objects) in KG
        :param b: int, the number of distinct edges (predicates) in KG
        :param n: int, the number of nodes in the subgraph
        :param e: int, the number of edges in the subgraph
        :param layers: array of neurons for the additional MLP
        :return: model
        """

        """ Currently supported 2 and 3. 3 as measured in the experiments """
        if matrix_mode == 0:
            # 2 matrices, one-hot encoding
            # create the MLP models
            nn1 = create_flatten_mlp((n, d), regress=False)
            nn2 = create_flatten_mlp((b, n, n), regress=False)
            # create the input to our final set of layers as the *output* of previous layer
            combinedInput = concatenate([nn1.output, nn2.output])
        elif matrix_mode == 1:
            # 2 matrices, binary encoding
            binary_d = int(np.ceil(np.log2(d))) + 1
            # create the MLP models
            nn1 = create_flatten_mlp((n, binary_d), regress=False)
            nn2 = create_flatten_mlp((b, n, n), regress=False)
            # create the input to our final set of layers as the *output* of previous layer
            combinedInput = concatenate([nn1.output, nn2.output])
        elif matrix_mode == 2:
            # 3 matrices, one-hot encoding
            # create the MLP models
            nn1 = create_flatten_mlp((n, d), include_layer=False, regress=False)
            nn2 = create_flatten_mlp((e, b), include_layer=False, regress=False)
            nn3 = create_flatten_mlp((e, n, n), include_layer=False, regress=False)
            # create the input to our final set of layers as the *output* of previous layer
            combinedInput = concatenate([nn1.output, nn2.output, nn3.output])
        elif matrix_mode == 3:
            # 2 matrices, binary encoding
            binary_d = int(np.ceil(np.log2(d))) + 1
            binary_b = int(np.ceil(np.log2(b))) + 1
            # create the MLP models
            nn1 = create_flatten_mlp((n, binary_d), False, regress=False)
            nn2 = create_flatten_mlp((e, binary_b), False, regress=False)
            nn3 = create_flatten_mlp((e, n, n), False, regress=False)
            # create the input to our final set of layers as the *output* of previous layer
            combinedInput = concatenate([nn1.output, nn2.output, nn3.output])
        elif matrix_mode == 4:
            # can be used for a single query type
            # create the MLP and CNN models
            binary_d = int(np.ceil(np.log2(d))) + 1
            binary_b = int(np.ceil(np.log2(b))) + 1
            # create the MLP models
            nn1 = create_flatten_mlp((n, binary_d), False, regress=False)
            nn2 = create_flatten_mlp((e, binary_b), False, regress=False)
            # create the input to our final set of layers as the *output* of previous layer
            combinedInput = concatenate([nn1.output, nn2.output])
        else:
            print("Non-supported")
            exit(1)

        # our final dense layers
        x = Dense(layers[0], activation="relu")(combinedInput)
        # x = Dropout(.1)(x)
        for i in range(1, len(layers)):
            x = Dense(layers[i], activation="relu")(x)
            # x = Dropout(.3)(x)
        x = Dense(1, activation="sigmoid")(x)

        if matrix_mode == 0 or matrix_mode == 1 or matrix_mode == 4:
            model = Model(inputs=[nn1.input, nn2.input], outputs=x)
        elif matrix_mode == 2 or matrix_mode == 3:
            model = Model(inputs=[nn1.input, nn2.input, nn3.input], outputs=x)
        else:
            print("Matrix mode unsupported")
            exit(1)
        return model




    def train_model(model, X, A, E, y, batch_size, epochs, matrix_mode, learning_rate = 1e-3,
                    decay = False, use_q_loss = True, scale = 0, do_training = False, encoding_avg_time = 0, sizes=None,
                    save_path:str = None, query_type=None, statistics_file_name:str = None, data: list = None,
                    avg_encoding_time_per_atom:float = None, n_atoms:int = None):

        assert save_path is not None
        assert query_type is not None
        assert statistics_file_name is not None
        assert data is not None
        assert avg_encoding_time_per_atom is not None
        assert n_atoms is not None
        """
        Actual training of the model
        :param model: the created model for training
        :param X: nodes matrix
        :param A: adjacency tensor
        :param E: edges matrix
        :param y: cardinality
        :param batch_size:
        :param epochs:
        :param matrix_mode:
        :param learning_rate:
        :param decay:
        :param use_q_loss:
        :param scale: type of scaling, recommended and best is the currently selected scale = 3
        :param do_training: to train or to predict
        :param encoding_avg_time: time needed for encoding, if needed for storing
        :return:
        """
        if decay:
            opt = Adam(lr = learning_rate, decay = learning_rate / 300)
        else:
            opt = Adam(lr = learning_rate)

        max_cardinality = max(y)
        min_cardinality = min(y)

        if do_training:
            write_MIN_MAX(query_type + "_min_max_" + dataset_name + ".txt", min(y), max(y))

        """ Other explored strategies, LMKG currently uses scale 3 """
        if scale == 0:
            print("We are using log1p")
            y = np.log(y + 1) + 1
        elif scale == 1:
            print("We are normalizing")
            y = normalize(y, min_cardinality, max_cardinality)
        elif scale == 2:
            scaler_y = MinMaxScaler()
            print("We are standardizing")
            y = np.log(y)
            scaler_y.fit(np.reshape(y, (-1, 1)))
            y = scaler_y.transform(np.reshape(y, (-1, 1)))
        elif scale == 3:
            if not do_training:
                global MIN, MAX
                """ We need the same min and max from the training data during evaluation """
                MIN, MAX = read_MIN_MAX(query_type+"_min_max_"+dataset_name+".txt")
                MIN = np.log(MIN)
                MAX = np.log(MAX)
            y = np.log(y)
            y = normalize_MINMAX(y)
        else:
            print("Scale not supported")

        """ As loss there is the option to have q_loss or mse, q_loss is currently better """
        print("As loss we are using qloss: " + str(use_q_loss))
        loss = q_loss if use_q_loss else 'mse',
        #loss = 'mse'
        metrics = ['mae', 'mse', 'accuracy', q_loss]
        # Model on cpu
        # with tf.device('/cpu:0'):
        model.compile(loss=loss, optimizer=opt, metrics = metrics)

        x=[]
        if matrix_mode == 0:
            x = [X, A]
        elif matrix_mode == 1:
            x = [X]
        elif matrix_mode == 2 or matrix_mode == 3:
            x = [X, E, A]
        elif matrix_mode == 4:
            x = [X, E]
        start_time = time.time()

        if do_training:
            # Initialize timing and metrics list
            epoch_times = []
            epoch_metrics = {}
            # LambdaCallback for timing
            time_callback = LambdaCallback(
                on_epoch_begin=lambda epoch, logs: epoch_times.append(time.time()),
                on_epoch_end=lambda epoch, logs: epoch_times.append(time.time())
            )

            history = model.fit(
                x=x, y=y,
                # validation_split=0.2,
                epochs=epochs, batch_size=batch_size, shuffle = True, callbacks=[time_callback])



        end_time = time.time() - start_time
        # How much time it took to train per atom over all episodes (ms)
        total_training_time_per_atom = end_time/n_atoms * 1000
        print_time = ('The time needed for training %.3f seconds \n' % end_time)
        if do_training:
            print(print_time)
            epoch_metrics = history.history
            epoch_durations = [epoch_times[i + 1] - epoch_times[i] for i in range(0, len(epoch_times) - 1, 2)]
            accumulated_times = np.cumsum(epoch_durations)

            epoch_dicts = []
            # Create dictionaries
            for i in range(epochs):
                epoch_dict = {'epoch': i + 1, 'duration': accumulated_times[i]}
                for metric, values in epoch_metrics.items():
                    epoch_dict[metric] = values[i]
                epoch_dicts.append(epoch_dict)
        else:
            epoch_dicts = None

        avg_time_prediction_ms = 0
        if do_training:
            preds = model.predict(x, batch_size=batch_size)
            pred_times = None
            total_pred_times = None
        else:
            pred_times = []
            total_pred_times = []
            for _ in range(1):
                for j in tqdm(range(0, len(y))):
                    start_time = time.time()
                    model([x[0][j:j+1], x[1][j:j+1], x[2][j:j+1]])
                    end_time = time.time() - start_time
                    pred_times.append(end_time * 1000) # Prediction time in ms
                    total_pred_times.append(end_time * 1000 + encoding_avg_time) # Prediction time in ms

                total_pred = 0
                predictions_start_time = time.time()
                # Predict the cardinality for the given triples
                preds = model.predict(x, batch_size=1)
                predictions_start_time = time.time() - predictions_start_time
                total_pred += predictions_start_time / len(x[0])
            avg_time_prediction_ms = (total_pred / 1) * 1000.0
            # Average Prediction per Query:
        print("The average prediction time is " + str(avg_time_prediction_ms))
        if not do_training:
            print("The average prediction time (custom) is " + str(np.average(pred_times)))
        print("The average encoding time is " + str(encoding_avg_time))
        print(" Total Time for Training and Encoding: ", str(avg_time_prediction_ms + encoding_avg_time))


        if scale == 0:
            preds = np.round(np.exp(preds-1)-1).astype(int)
            y = np.round(np.exp(y-1)-1).astype(int)
        elif scale == 1:
            preds = denormalize(preds, min_cardinality, max_cardinality)
            y = denormalize(y, min_cardinality, max_cardinality)
        elif scale == 2:
            preds = np.reshape(preds, (-1, 1))
            y = np.reshape(y, (-1, 1))
            preds = scaler_y.inverse_transform(preds)
            y = scaler_y.inverse_transform(y)
            preds = np.exp(preds)
            y = np.exp(y)
        elif scale == 3:
            preds = denormalize_MINMAX(preds)
            y = denormalize_MINMAX(y)
            preds = np.exp(preds)
            y = np.exp(y)
        else:
            print("Scale not supported")

        # removing outliers if needed
        nb_outliers = 0
        if nb_outliers > 0:
            sorted_y = y
            sorted_y = sorted(sorted_y, reverse = True)
            outliers = sorted_y[:nb_outliers]
            preds_outlier = []
            y_outlier = []
            for i in range(len(y)):
                if y[i] not in outliers:
                    y_outlier.append(y[i])
                    preds_outlier.append(preds[i])
            y = y_outlier
            preds = preds_outlier

        # accuracy statistics for the network

        if do_training:
            # Saving the training times:
            pass
            # training_timing = {'avg_encoding_time_per_atom': avg_encoding_time_per_atom,
            #                    'total_training_time_per_atom': total_training_time_per_atom,
            #                    'epochs': epochs}
            # with open(f"{save_path}/training_timing.json", 'w') as file:
            #     json.dump(training_timing, file, indent=4)

        correct_estimate = check_network_estimates(preds, y, statistics_file_name = statistics_file_name,
                                                   print_time=print_time, nb_outliers = nb_outliers, sizes=sizes,
                                                   pred_times=pred_times, save_path=save_path, data=data,
                                                   training_progress=epoch_dicts, total_pred_times=total_pred_times),
        # correct_estimate = store_statistics.check_network_estimates(preds, y, statistics_file_name = statistics_file_name, print_time=print_time, nb_outliers = nb_outliers)

        return correct_estimate


def run_lmkg(dataset: str, query_form: str, eval_folder: str, query_filename: str, train: bool = True,
             inductive:str = 'false', DATASETPATH=None):

    assert DATASETPATH is not None


    Path(f"{DATASETPATH}{dataset}/Results/{eval_folder}/LMKG").mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--training', dest='training', action='store_true')
    parser.add_argument('--eval', dest='training', action='store_false')
    parser.set_defaults(training=True)
    parser.add_argument('--batch-size', type=int, default=1, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=20, help='# epochs')
    parser.add_argument('--dataset', type=str, default='swdf', help='The dataset name')
    parser.add_argument('--query-type', type=str, default='combined', help='Supports star, chain, combined or complex')
    parser.add_argument('--datasets-path', type=str, default='final_datasets',
                        help='Path to the train and eval queries')
    parser.add_argument('--query-join', type=int, default=3, help='The size of the join.')
    parser.add_argument('--layers', type=int, default=2, help='The number of layers in the NN.')
    parser.add_argument('--neurons', type=int, default=256, help='The number of neurons in each of the layers.')
    parser.add_argument('--decay', action="store_true", help='Decay of the learning rate.')
    parser.add_argument('--test-mode', type=str, default="all",
                        help='[all, star, chain], Used for testing. In case we have trained a combined model (for star and chain) '
                             'we can evaluate only on star, only on chain or on all')

    args = parser.parse_args()

    print("GPU AVAILABLE")
    print(tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    ''' parameters for the network'''
    # Fixed parameters
    matrix_mode = 3
    train_tuples = 1_000_000_000
    # Changable parameters
    # do_training = args.training
    do_training = train
    create_new_model = True
    if not do_training:
        create_new_model = False
    batch_size = args.batch_size
    epochs = args.epochs
    datasets_path = args.datasets_path
    decay = args.decay

    '''dimension parameters for the network'''
    global dataset_name
    # dataset_name = args.dataset
    dataset_name = dataset
    if (dataset_name == 'swdf') or (dataset_name == 'swdf_inductive'):
        # SWDF
        d = 76712
        b = 171
        n = 11
        e = 10
    elif dataset_name == 'yago':
        # YAGO
        d = 13000179
        b = 92
        n = 10
        e = 9
    elif dataset_name == 'lubm':
        # LUBM1M
        d = 664050
        b = 19
        n = 10
        e = 9
    elif dataset_name == "freebase":
        print("The dataset is not an option, therefore D B N E need to be set properly")
        d = 14505
        b = 237
        n = 3
        e = 2

    # YAGO::
    # elif dataset_name == "custom":
    #     d = 13000179
    #     b = 92
    #     n = 10
    #     e = 9

    # :
    elif dataset_name == "custom":
        d = 5000000
        b = 872
        n = 9
        e = 8
    elif dataset_name == "wikidata":
        d = 5000000
        b = 872
        n = 9
        e = 8
    #     #exit(1)

    layers = [args.neurons for i in range(args.layers)]
    nn_size_name = '_'.join([str(neuron) for neuron in layers])
    scale = 3

    query_join = args.query_join
    # Can be [star chain - combined (query-size grouping) allstar allchain (query-type grouping) alltypessizes (single model), complex (single model)]
    #query_type = args.query_type
    # query_type = "star"
    if query_form == "path":
        query_type = 'chain'
    else:
        query_type = query_form

    print(" The model is trained: %r \n Batch size: %s "
          "\n Epochs: %d \n Dataset: %s \n Query type: %s \n "
          "Datasets path: %s \n Query Join: %d \n NN Size: %s \n"
          "Learning rate decay: %r \n"
          % (do_training, batch_size, epochs, dataset_name, query_type, datasets_path, query_join, nn_size_name, decay))

    #if "complex" not in query_type and "all" not in query_type:
    #    query_type = query_type + '_' + str(query_join)

    if query_join >= 5:
        n = 6
        e = 5
    if query_join >= 8:
        n = 9
        e = 8

    if "complex" in query_type:
        pass
        #n = 8
        #e = 6

    model_name = 'LMKGS-' + query_type + '-q_loss-' + nn_size_name + '-' + dataset_name + 'scale-' + str(
        scale) + '_decay' + str(decay) + '.h5'

    weights_name = model_name
    statistics_file_name = 'statistics-' + model_name

    if do_training:
        if "star" in query_type and 'all' not in query_type:
            print("entering for star queries")
            file_name = datasets_path + '/generated_' + dataset_name + '/star/' + query_type + '.txt'
        elif 'chain' in query_type and 'all' not in query_type:
            print("entering for chain queries")
            file_name = datasets_path + '/generated_' + dataset_name + '/path/' + query_type + '.txt'
        elif 'combined' in query_type:
            print("entering for query size grouping")
            file_name_star = datasets_path + '/generated_' + dataset_name + '/star/star_' + str(query_join) + '.txt'
            file_name_chain = datasets_path + '/generated_' + dataset_name + '/path/chain_' + str(query_join) + '.txt'
        elif 'allstar' in query_type or 'allchain' in query_type or 'alltypessizes' in query_type:
            all_file_names = []
            num_joins = [2]  # ,3,5,8]
            for i in num_joins:
                if 'allstar' in query_type or 'alltypessizes' in query_type:
                    all_file_names.append(
                        datasets_path + '/generated_' + dataset_name + '/star/star_' + str(i) + '.txt')
                if 'allchain' in query_type or 'alltypessizes' in query_type:
                    all_file_names.append(
                        datasets_path + '/generated_' + dataset_name + '/path/chain_' + str(i) + '.txt')
        elif 'complex' in query_type:
            path = datasets_path + '/generated_' + dataset_name + '/complex/'
            complex_file_names = []
            complex_file_names.append(path + 'swdf_complex_star1_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star2_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain3.txt')
        else:
            print("Query type not supported.")
            exit(1)
    else:
        if "star" in query_type and 'all' not in query_type:
            print("entering for star")
            file_name = datasets_path + '/generated_' + dataset_name + '/star/eval_600_' + query_type + '.txt'
        elif 'chain' in query_type and 'all' not in query_type:
            print("entering for chain")
            file_name = datasets_path + '/generated_' + dataset_name + '/path/eval_600_' + query_type + '.txt'
        elif 'combined' in query_type:
            print("entering for combined")
            file_name_star = datasets_path + '/generated_' + dataset_name + '/star/eval_600_star_' + str(
                query_join) + '.txt'
            file_name_chain = datasets_path + '/generated_' + dataset_name + '/path/eval_600_chain_' + str(
                query_join) + '.txt'
        elif 'allstar' in query_type or 'allchain' in query_type or 'alltypessizes' in query_type:
            all_file_names = []
            num_joins = [2]  # ,3,5,8]
            for i in num_joins:
                if 'allstar' in query_type or 'alltypessizes' in query_type:
                    all_file_names.append(
                        datasets_path + '/generated_' + dataset_name + '/star/eval_600_star_' + str(i) + '.txt')
                if 'allchain' in query_type or 'alltypessizes' in query_type:
                    all_file_names.append(
                        datasets_path + '/generated_' + dataset_name + '/path/eval_600_chain_' + str(i) + '.txt')
        elif 'complex' in query_type:
            path = datasets_path + '/generated_' + dataset_name + '/complex/'
            complex_file_names = []
            complex_file_names.append(path + 'swdf_complex_star1_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star2_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain3.txt')
        else:
            print("Query type not supported.")
            exit(1)

    if inductive == 'false':
        file_name = Path(f"{DATASETPATH}{dataset}/{query_form}/{query_filename}")
    elif inductive == 'full':
        file_name = (Path(f"{DATASETPATH}{dataset}/{query_form}/disjoint_train.json"),
                     Path(f"{DATASETPATH}{dataset}/{query_form}/disjoint_test.json"))

    if "star" in query_type or "complex" in query_type and 'all' not in query_type:
        # X, A, E, y, encoding_time = complex_reader.read_queries(file_name, d, b, n, e, query_type ="star")
        X, A, E, y, encoding_time, avg_encoding_time_per_atom, sizes, data, n_atoms = complex_reader.custom_reader(file_name, d, b, n, e, dataset=dataset_name,
                                                                              train=train, inductive=inductive, DATASETPATH=DATASETPATH)

        # X, A, E, y, encoding_time = read_star_graph_pattern(d, b, n, e, file_name, train_tuples, matrix_mode, 0)
    elif "chain" in query_type and 'all' not in query_type:
        # X, A, E, y, encoding_time = complex_reader.read_queries(file_name, d, b, n, e, query_type ="chain")
        X, A, E, y, encoding_time, avg_encoding_time_per_atom, sizes, data, n_atoms = complex_reader.custom_reader(file_name, d, b, n, e, dataset=dataset_name,
                                                                              train=train, inductive=inductive, DATASETPATH=DATASETPATH)

        # X, A, E, y, encoding_time = read_chain_graph_pattern(d, b, n, e, file_name, train_tuples, matrix_mode, 0)
    elif 'combined' in query_type:
        test_mode = args.test_mode
        statistics_file_name += test_mode
        X, A, E, y, encoding_time, sizes = complex_reader.read_combined(d, b, n, e, file_name_star, file_name_chain,
                                                                        train_tuples, test_mode)
    elif 'allstar' in query_type or 'allchain' in query_type or 'alltypessizes' in query_type:
        X, A, E, y, encoding_time, sizes = complex_reader.read_combined_all_sizes_star_or_chain(d, b, n, e,
                                                                                                all_file_names,
                                                                                                train_tuples)
    elif 'complex' in query_type:
        if not do_training:
            X, A, E, y, encoding_time = complex_reader.read_complex_queries(complex_file_names, d, b, n, e)
        else:
            X, A, E, y, encoding_time = complex_reader.read_complex_queries(complex_file_names, d, b, n, e)
    else:
        print("Query type not supported.")
        exit(1)

    avg_time_encoding_ms = (encoding_time / len(y)) * 1000.0

    '''training/testing/predicting'''
    '''create the model'''
    print("Creating Model..")

    save_path = Path(f"{DATASETPATH}{dataset}/Results/{eval_folder}/LMKG/")

    if create_new_model:
        '''create new model for the data'''
        nn_model = create_model(matrix_mode, d, b, n, e, layers=layers)
        print(nn_model.summary())

    else:
        # load the model
        #save_path = 'models/' + dataset_name + "/"

        print("Loading model: ", save_path / model_name)
        nn_model = load_model(save_path / model_name, custom_objects={"q_loss": q_loss})
        print(nn_model.summary())

    '''train the model'''
    print("Starting Training..")

    train_model(nn_model, X, A, E, y, batch_size, epochs, decay=decay, matrix_mode=matrix_mode,
                scale=scale, do_training=do_training, encoding_avg_time=avg_time_encoding_ms, sizes=sizes,
                save_path=save_path, query_type=query_type ,statistics_file_name=statistics_file_name, data=data,
                avg_encoding_time_per_atom=avg_encoding_time_per_atom,
                n_atoms=n_atoms)

    '''storing of the model'''
    if do_training:


        if not os.path.exists(save_path):
            os.makedirs(save_path)
        nn_model.save(save_path / model_name)

        '''storing just the weights of the model'''
        weights = nn_model.get_weights()
        with open(Path(f"{save_path}/' + model_name + '_size.txt"), "w") as f:
            f.write(str(weights))

    return n_atoms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', dest='training', action='store_true')
    parser.add_argument('--eval', dest='training', action='store_false')
    parser.set_defaults(training=True)
    parser.add_argument('--batch-size', type=int, default=512, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=100, help='# epochs')
    parser.add_argument('--dataset', type=str, default='swdf', help='The dataset name')
    parser.add_argument('--query-type', type=str, default ='combined', help='Supports star, chain, combined or complex')
    parser.add_argument('--datasets-path', type=str, default='final_datasets', help='Path to the train and eval queries')
    parser.add_argument('--query-join', type=int, default=3, help='The size of the join.')
    parser.add_argument('--layers', type=int, default=2, help='The number of layers in the NN.')
    parser.add_argument('--neurons', type=int, default=256, help='The number of neurons in each of the layers.')
    parser.add_argument('--decay', action="store_true", help='Decay of the learning rate.')
    parser.add_argument('--test-mode', type=str, default = "all", help='[all, star, chain], Used for testing. In case we have trained a combined model (for star and chain) '
                                                        'we can evaluate only on star, only on chain or on all')

    args = parser.parse_args()

    print("GPU AVAILABLE")
    print(tf.test.is_gpu_available())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    ''' parameters for the network'''
    # Fixed parameters
    matrix_mode = 3
    train_tuples = 1_000_000_000
    # Changable parameters
    # do_training = args.training
    do_training = False
    create_new_model = True
    if not do_training:
        create_new_model = False
    batch_size = args.batch_size
    epochs = args.epochs
    datasets_path = args.datasets_path
    decay = args.decay


    '''dimension parameters for the network'''
    global dataset_name
    #dataset_name = args.dataset
    dataset_name = "custom"
    if dataset_name == 'swdf':
        # SWDF
        d = 76712
        b = 171
        n = 11
        e = 10
    elif dataset_name == 'yago':
        # YAGO
        d = 13000179
        b = 92
        n = 10
        e = 9
    elif dataset_name == 'lubm':
        # LUBM1M
        d = 664050
        b = 19
        n = 4
        e = 4
    elif dataset_name == "freebase":
        print("The dataset is not an option, therefore D B N E need to be set properly")
        d = 14505
        b = 237
        n = 3
        e = 2

    # YAGO::
    # elif dataset_name == "custom":
    #     d = 13000179
    #     b = 92
    #     n = 10
    #     e = 9

    # :
    elif dataset_name == "custom":
        d = 5000000
        b = 872
        n = 9
        e = 8
    #     #exit(1)

    layers = [args.neurons for i in range(args.layers)]
    nn_size_name = '_'.join([str(neuron) for neuron in layers])
    scale = 3

    query_join = args.query_join
    # Can be [star chain - combined (query-size grouping) allstar allchain (query-type grouping) alltypessizes (single model), complex (single model)]
    query_type = args.query_type
    #query_type = "star"
    query_type = 'chain'



    print(" The model is trained: %r \n Batch size: %s "
          "\n Epochs: %d \n Dataset: %s \n Query type: %s \n "
          "Datasets path: %s \n Query Join: %d \n NN Size: %s \n"
          "Learning rate decay: %r \n"
          % (do_training, batch_size, epochs, dataset_name, query_type, datasets_path, query_join, nn_size_name, decay))

    if "complex" not in query_type and "all" not in query_type:
        query_type = query_type + '_' + str(query_join)


    if query_join >= 5:
        n = 6
        e = 5
    if query_join >= 8:
        n = 9
        e = 8

    if "complex" in query_type:
        n = 8
        e = 6

    model_name = 'LMKGS-' + query_type + '-q_loss-' + nn_size_name + '-' + dataset_name + 'scale-' + str(scale) + '_decay'+str(decay) + '.h5'

    weights_name = model_name
    statistics_file_name = 'statistics-'+ model_name

    if do_training:
        if "star" in query_type and 'all' not in query_type:
            print("entering for star queries")
            file_name = datasets_path + '/generated_' + dataset_name + '/star/'+query_type+'.txt'
        elif 'chain' in query_type and 'all' not in query_type:
            print("entering for chain queries")
            file_name = datasets_path + '/generated_' + dataset_name + '/path/' + query_type + '.txt'
        elif 'combined' in query_type:
            print("entering for query size grouping")
            file_name_star = datasets_path + '/generated_' + dataset_name + '/star/star_' + str(query_join) + '.txt'
            file_name_chain = datasets_path + '/generated_' + dataset_name + '/path/chain_' + str(query_join) + '.txt'
        elif 'allstar' in query_type or 'allchain' in query_type or 'alltypessizes' in query_type:
            all_file_names = []
            num_joins = [2]#,3,5,8]
            for i in num_joins:
                if 'allstar' in query_type or 'alltypessizes' in query_type:
                    all_file_names.append(datasets_path + '/generated_' + dataset_name + '/star/star_' + str(i) + '.txt')
                if 'allchain' in query_type or 'alltypessizes' in query_type:
                    all_file_names.append(datasets_path + '/generated_' + dataset_name + '/path/chain_' + str(i) + '.txt')
        elif 'complex' in query_type:
            path = datasets_path + '/generated_' + dataset_name + '/complex/'
            complex_file_names = []
            complex_file_names.append(path + 'swdf_complex_star1_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star2_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain3.txt')
        else:
            print("Query type not supported.")
            exit(1)
    else:
        if "star" in query_type and 'all' not in query_type:
            print("entering for star")
            file_name = datasets_path + '/generated_' + dataset_name + '/star/eval_600_'+query_type+'.txt'
        elif 'chain' in query_type and 'all' not in query_type:
            print("entering for chain")
            file_name = datasets_path + '/generated_' + dataset_name + '/path/eval_600_'+query_type+'.txt'
        elif 'combined' in query_type:
            print("entering for combined")
            file_name_star = datasets_path + '/generated_' + dataset_name + '/star/eval_600_star_' + str(query_join) + '.txt'
            file_name_chain = datasets_path + '/generated_' + dataset_name + '/path/eval_600_chain_' + str(query_join) + '.txt'
        elif 'allstar' in query_type or 'allchain' in query_type or 'alltypessizes' in query_type:
            all_file_names = []
            num_joins = [2]#,3,5,8]
            for i in num_joins:
                if 'allstar' in query_type or 'alltypessizes' in query_type:
                    all_file_names.append(datasets_path + '/generated_' + dataset_name + '/star/eval_600_star_' + str(i) + '.txt')
                if 'allchain' in query_type or 'alltypessizes' in query_type:
                    all_file_names.append(datasets_path + '/generated_' + dataset_name + '/path/eval_600_chain_' + str(i) + '.txt')
        elif 'complex' in query_type:
            path = datasets_path + '/generated_' + dataset_name + '/complex/'
            complex_file_names = []
            complex_file_names.append(path + 'swdf_complex_star1_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star2_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain2.txt')
            complex_file_names.append(path + 'swdf_complex_star3_chain3.txt')
        else:
            print("Query type not supported.")
            exit(1)

    file_name = "final_datasets/freebase_train_test_data_2tp_star.json"
    if "star" in query_type and 'all' not in query_type:
        #X, A, E, y, encoding_time = complex_reader.read_queries(file_name, d, b, n, e, query_type ="star")
        X, A, E, y, encoding_time, sizes = complex_reader.custom_reader(file_name, d, b, n, e)

        # X, A, E, y, encoding_time = read_star_graph_pattern(d, b, n, e, file_name, train_tuples, matrix_mode, 0)
    elif "chain" in query_type and 'all' not in query_type:
        #X, A, E, y, encoding_time = complex_reader.read_queries(file_name, d, b, n, e, query_type ="chain")
        X, A, E, y, encoding_time, sizes = complex_reader.custom_reader(file_name, d, b, n, e)

        # X, A, E, y, encoding_time = read_chain_graph_pattern(d, b, n, e, file_name, train_tuples, matrix_mode, 0)
    elif 'combined' in query_type:
        test_mode = args.test_mode
        statistics_file_name += test_mode
        X, A, E, y, encoding_time, sizes = complex_reader.read_combined(d, b, n, e, file_name_star, file_name_chain, train_tuples, test_mode)
    elif 'allstar' in query_type or 'allchain' in query_type or 'alltypessizes' in query_type:
        X, A, E, y, encoding_time, sizes = complex_reader.read_combined_all_sizes_star_or_chain(d, b, n, e, all_file_names, train_tuples)
    elif 'complex' in query_type:
        if not do_training:
            X, A, E, y, encoding_time = complex_reader.read_complex_queries(complex_file_names, d, b, n, e, 100)
        else:
            X, A, E, y, encoding_time = complex_reader.read_complex_queries(complex_file_names, d, b, n, e)
    else:
        print("Query type not supported.")
        exit(1)

    avg_time_encoding_ms = (encoding_time / len(y)) * 1000.0


    '''training/testing/predicting'''
    '''create the model'''
    print("Creating Model..")

    if create_new_model:
        '''create new model for the data'''
        nn_model = create_model(matrix_mode, d, b, n, e, layers = layers)
        print(nn_model.summary())

    else:
        #load the model
        save_path = 'models/'+dataset_name+"/"
        print("Loading model: ", save_path + model_name)
        nn_model = load_model(save_path + model_name, custom_objects={"q_loss": q_loss})
        print(nn_model.summary())


    '''train the model'''
    print("Starting Training..")
    train_model(nn_model, X, A, E, y, batch_size, epochs, decay = decay, matrix_mode = matrix_mode,
                scale = scale, do_training=do_training, encoding_avg_time=avg_time_encoding_ms, sizes=sizes)


    '''storing of the model'''
    if do_training:
        save_path = 'models/'+dataset_name+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        nn_model.save(save_path + model_name)

        '''storing just the weights of the model'''
        weights = nn_model.get_weights()
        with open('weights/' + model_name + '_size.txt', "w") as f:
            f.write(str(weights))