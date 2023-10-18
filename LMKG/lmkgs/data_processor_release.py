import numpy as np
import random
import time

def binary_representation_from_int(number, max_size=16):
    """ Returns a binary number from an integer with bits set to max size """
    return format(number,'b').zfill(max_size)

def numpy_binary(number, max_size = 16):
    """
    Returns a binary representation of a given number
    :param number: needs conversion to binary representation
    :param max_size: max number of bits of the binary representation
    :return: binary representation
    """
    binary_str = binary_representation_from_int(number, max_size)
    arr = list(map(int, binary_str))
    return np.array(arr)

def read_star_graph_pattern(d, b, n, e, file_name, train_tuples = 10000000, matrix_mode = 0, star_mode = 0):
    """
    Creates encoding from queries in the file
    :param d: int, the number of distinct nodes (subjects + objects) in KG
    :param b: int, the number of distinct edges (predicates) in KG
    :param n: int, the number of nodes in the subgraph
    :param e: int, the number of edges in the subgraph
    :param file_name: str, the input file name of the queries used for training or testing
    :param train_tuples: int, limits the training sample size, in default we train on all of them
    :param matrix_mode: 0, 1 being two matrices 2 and 3 being with 3 matrices as reported in paper, one-hot and binary encoding
    :param star_mode:
    :return: encoded queries
    """

    print("started reading and creating star patterns")
    star_node_id = 0
    star_predicate_id = 0
    # we need to add one node for star
    X=[]
    A=[]
    E=[]
    y=[]
    total_predicate_star = 0
    with open(file_name) as fp:
        line_nb = 0
        lines = fp.readlines()
        # We need the shuffle in cases where we want to take a subset of the data
        # random.shuffle(lines)
        # for line in lines:
        time_start = time.time()
        for line in lines:
            if line_nb == train_tuples:
                print("We are going out due to training size limitation")
                break
            patterns, cardinality = line.split(":")
            cardinality = int(cardinality)

            if "*-" in patterns:
                total_predicate_star += 1

            if (line_nb % 100000) == 0:
                print(line_nb)
            line_nb += 1
            patterns = patterns.split(",")
            if matrix_mode == 2:
                x = np.zeros((n, d), dtype='uint8')
                ep = np.zeros((e, b), dtype='uint8')
                a = np.zeros((e, n, n), dtype='uint8')
            elif matrix_mode == 3:
                bits_d = int(np.ceil(np.log2(d))) + 1
                bits_b = int(np.ceil(np.log2(b))) + 1
                x = np.zeros((n, bits_d), dtype='uint8')
                ep = np.zeros((e, bits_b), dtype='uint8')
                a = np.zeros((e, n, n), dtype='uint8')
            else:
                print("Matrix mode unspecified")
                exit(1)
            nodes = set()
            # Only the subject will be added as a star node
            nodes.add("*")
            total_zero = 0
            predicates = set()
            pattern_zero_flag = False
            predicate_node_counter = 1
            for pattern in patterns:
                predicate, object = pattern.split("-")
                if object != "*":
                    if int(object) == 0:
                        pattern_zero_flag = True
                    else:
                        nodes.add(object)
                else:
                    nodes.add("*n"+str(predicate))
                    star_node_id += 1

                if pattern_zero_flag:
                    break
                if predicate != "*":
                    if int(predicate) == 0:
                        pattern_zero_flag = True
                    else:
                        predicates.add(predicate)
                else:
                    predicates.add("*p"+str(object))
                    predicate_node_counter += 1

                if pattern_zero_flag:
                    break
            if pattern_zero_flag:
                continue
            y.append(cardinality)
            if pattern_zero_flag:
                continue

            if total_zero > 0:
                print("Total zero "+str(total_zero))
                exit(1)

            nodes=list(nodes)
            nodes.sort()
            id = 0
            nodes_dict = dict()
            for node in nodes:
                nodes_dict[node]=id
                id += 1

            predicates = list(predicates)
            predicates.sort()
            id = 0
            predicates_dict = dict()
            for predicate in predicates:
                predicates_dict[predicate] = id
                id += 1

            # creating the encodings
            for pattern in patterns:
                predicate, object = pattern.split("-")
                subject_id = "*"
                object_id = ""
                predicate_id = ""
                if object != "*":
                    object_id = object
                else:
                    object_id = "*n" + str(predicate)

                if predicate != "*":
                    predicate_id = predicate
                else:
                    predicate_id = "*p" + str(object)


                if matrix_mode == 2:
                    subject_subgraph_id = nodes_dict[subject_id]
                    predicate_subgraph_id = predicates_dict[predicate_id]
                    object_subgraph_id = nodes_dict[object_id]
                    a[predicate_subgraph_id, subject_subgraph_id, object_subgraph_id] = 1

                    if predicate_id == star_predicate_id and star_mode == 1:
                        for pred_id in range(b):
                            ep[predicate_subgraph_id, predicate_id] = 1
                    else:
                        ep[predicate_subgraph_id, predicate_id] = 1

                    if subject_id == star_node_id and star_mode == 1:
                        for subj_id in range(d):
                            x[subject_subgraph_id, subj_id] = 1
                    else:
                        x[subject_subgraph_id, subject_id] = 1

                    if object_id == star_node_id and star_mode == 1:
                        for obj_id in range(d):
                            x[object_subgraph_id, obj_id] = 1
                    else:
                        x[object_subgraph_id, object_id] = 1

                elif matrix_mode == 3:

                    subject_subgraph_id = nodes_dict[subject_id]
                    predicate_subgraph_id = predicates_dict[predicate_id]
                    object_subgraph_id = nodes_dict[object_id]
                    a[predicate_subgraph_id, subject_subgraph_id, object_subgraph_id] = 1

                    predicate_nb = 0
                    if "*" not in predicate_id:
                        predicate_nb = int(predicate_id)
                    arr = numpy_binary(predicate_nb, bits_b)
                    for i in range(len(arr)):
                        ep[predicate_subgraph_id][i] = arr[i]

                    subject_nb = 0
                    if "*" not in subject_id:
                        subject_nb = int(subject_id)
                    arr = numpy_binary(subject_nb, bits_d)
                    for i in range(len(arr)):
                        x[subject_subgraph_id][i] = arr[i]

                    object_nb = 0
                    if "*" not in object_id:
                        object_nb = int(object_id)
                    arr = numpy_binary(object_nb, bits_d)

                    for i in range(len(arr)):
                        x[object_subgraph_id][i] = arr[i]

                else:
                    print("Matrix mode unspecified")
                    exit(1)
            X.append(x)
            E.append(ep)
            A.append(a)

    time_end = time.time() - time_start
    return np.array(X),np.array(A),np.array(E), np.array(y), time_end


def read_chain_graph_pattern(d, b, n, e, file_name, train_tuples = 10000, matrix_mode = 0, star_mode = 0):
    """
    Creates encoding from queries in the file
    :param d: int, the number of distinct nodes (subjects + objects) in KG
    :param b: int, the number of distinct edges (predicates) in KG
    :param n: int, the number of nodes in the subgraph
    :param e: int, the number of edges in the subgraph
    :param file_name: str, the input file name of the queries used for training or testing
    :param train_tuples: int, limits the training sample size, in default we train on all of them
    :param matrix_mode: 0, 1 being two matrices 2 and 3 being with 3 matrices as reported in paper, one-hot and binary encoding
    :param star_mode:
    :return: encoded queries
    """
    print("started reading and creating chain patterns")
    X=[]
    A=[]
    E=[]
    y=[]
    total_predicate_star = 0
    with open(file_name) as fp:
        line_nb = 0
        lines = fp.readlines()
        random.shuffle(lines)
        time_start = time.time()
        for line in lines:
            if line_nb == train_tuples:
                print("We are going out due to training size limitation")
                break

            patterns, cardinality = line.split(",")
            cardinality = int(cardinality)

            if (line_nb % 100000) == 0:
                print(line_nb)
            line_nb += 1

            if matrix_mode == 2:
                x = np.zeros((n, d), dtype='uint8')
                ep = np.zeros((e, b), dtype='uint8')
                a = np.zeros((e, n, n), dtype='uint8')
            elif matrix_mode == 3:
                bits_d = int(np.ceil(np.log2(d))) + 1
                bits_b = int(np.ceil(np.log2(b))) + 1
                x = np.zeros((n, bits_d), dtype='uint8')
                ep = np.zeros((e, bits_b), dtype='uint8')
                a = np.zeros((e, n, n), dtype='uint8')
            else:
                print("Matrix mode unspecified")
                exit(1)

            nodes = list()
            total_zero = 0
            predicates = list()
            pattern_zero_flag = False

            star_node_counter = 1
            predicate_node_counter = 1

            patterns1 = patterns.split("-")
            first_time = True
            patterns = []
            i = 0
            nb_predicate_star = 0
            while i < len(patterns1) - 1:
                if first_time:
                    s = patterns1[i]
                    p = patterns1[i + 1]
                    o = patterns1[i + 2]
                else:
                    s = o
                    p = patterns1[i + 1]
                    o = patterns1[i + 2]

                if first_time:
                    if "*" not in s:
                        nodes.append(s)
                    else:
                        s = s + "00n" + str(p)
                        nodes.append(s)
                if "*" not in o:
                    nodes.append(o)
                else:
                    if i == len(patterns1) - 3:
                        o = o + "xn" + str(p)
                    else:
                        o = o + str(nb_predicate_star) + "n" + str(p)
                        nb_predicate_star += 1
                    nodes.append(o)

                if "*" not in p:
                    predicates.append(p)
                else:
                    # omitting predicates star
                    total_predicate_star += 1
                    p = p + "p" + str(s)
                    predicates.append(p)
                patterns.append(s + "-" + p + "-" + o)
                i += 2
                first_time = False


            y.append(cardinality)


            if total_zero > 0:
                print("Total zero "+str(total_zero))
                exit(1)

            nodes_list = []
            [nodes_list.append(x) for x in nodes if x not in nodes_list]
            # nodes.sort()
            nodes = list(set(nodes))
            id = 0
            nodes_dict = dict()
            for node in nodes_list:
                nodes_dict[node]=id
                id += 1


            predicates_list = []
            [predicates_list.append(x) for x in predicates if x not in predicates_list]
            # predicates.sort()
            id = 0
            predicates_dict = dict()
            for predicate in predicates_list:
                predicates_dict[predicate] = id
                id += 1


            # creating the encodings
            for pattern in patterns:
                subject, predicate, object = pattern.split("-")
                subject_id = subject
                object_id = object
                predicate_id = predicate


                if matrix_mode == 2:
                    subject_subgraph_id = nodes_dict[subject_id]
                    predicate_subgraph_id = predicates_dict[predicate_id]
                    object_subgraph_id = nodes_dict[object_id]
                    a[predicate_subgraph_id, subject_subgraph_id, object_subgraph_id] = 1
                    ep[predicate_subgraph_id, predicate_id] = 1
                    x[subject_subgraph_id, subject_id] = 1
                    x[object_subgraph_id, object_id] = 1

                elif matrix_mode == 3:
                    subject_subgraph_id = nodes_dict[subject_id]
                    predicate_subgraph_id = predicates_dict[predicate_id]
                    object_subgraph_id = nodes_dict[object_id]
                    a[predicate_subgraph_id, subject_subgraph_id, object_subgraph_id] = 1

                    predicate_nb = 0
                    if "*" not in predicate_id:
                        predicate_nb = int(predicate_id)
                    arr = numpy_binary(predicate_nb, bits_b)

                    for i in range(len(arr)):
                        ep[predicate_subgraph_id][i] = arr[i]

                    subject_nb = 0
                    if "*" not in subject_id:
                        subject_nb = int(subject_id)
                    arr = numpy_binary(subject_nb, bits_d)
                    for i in range(len(arr)):
                        x[subject_subgraph_id][i] = arr[i]

                    object_nb = 0
                    if "*" not in object_id:
                        object_nb = int(object_id)
                    arr = numpy_binary(object_nb, bits_d)
                    for i in range(len(arr)):
                        x[object_subgraph_id][i] = arr[i]

                else:
                    print("Matrix mode unspecified")
                    exit(1)
            X.append(x)
            E.append(ep)
            A.append(a)


    time_end = time.time() - time_start
    return np.array(X),np.array(A),np.array(E), np.array(y), time_end


def read_combined(d, b, n, e, file_name_star, file_name_chain, train_tuples = 10000, matrix_mode = 0, star_mode = 0, test_mode = "star"):
    """
    Creates encoding from queries in the file
    :param d: int, the number of distinct nodes (subjects + objects) in KG
    :param b: int, the number of distinct edges (predicates) in KG
    :param n: int, the number of nodes in the subgraph
    :param e: int, the number of edges in the subgraph
    :param file_name_star: str, the input file name of the queries used for training or testing
    :param file_name_chain: str, the input file name of the queries used for training or testing
    :param train_tuples: int, limits the training sample size, in default we train on all of them
    :param matrix_mode: 0, 1 being two matrices 2 and 3 being with 3 matrices as reported in paper, one-hot and binary encoding
    :param star_mode:
    :param test_mode: str, it can be star or chain
    :return: encoded queries
    """


    if "star" in test_mode:
        return read_star_graph_pattern(d, b, n, e, file_name_star, train_tuples, matrix_mode)
    if "chain" in test_mode:
        return read_chain_graph_pattern(d, b, n, e , file_name_chain, train_tuples, matrix_mode)

    X_s, A_s, E_s, y_s, time_start_star = read_star_graph_pattern(d, b, n, e, file_name_star, train_tuples, matrix_mode)
    X_c, A_c, E_c, y_c, time_start_chain = read_chain_graph_pattern(d, b, n, e , file_name_chain, train_tuples, matrix_mode)
    X = np.concatenate([X_s, X_c], axis = 0)
    A = np.concatenate([A_s, A_c], axis = 0)
    E = np.concatenate([E_s, E_c], axis = 0)
    y = np.concatenate([y_s, y_c], axis = 0)

    return X, A, E, y, (time_start_star + time_start_chain)


def read_combined_all_sizes_star_or_chain(d, b, n, e, file_names, train_tuples = 10000, matrix_mode = 0, star_mode = 0, test_mode = "star"):
    """
    Creates encoding from queries in the file
    :param d: int, the number of distinct nodes (subjects + objects) in KG
    :param b: int, the number of distinct edges (predicates) in KG
    :param n: int, the number of nodes in the subgraph
    :param e: int, the number of edges in the subgraph
    :param file_names: str, the input file name of the queries used for training or testing
    :param train_tuples: int, limits the training sample size, in default we train on all of them
    :param matrix_mode: 0, 1 being two matrices 2 and 3 being with 3 matrices as reported in paper, one-hot and binary encoding
    :param star_mode:
    :param test_mode: str, it can be star or chain
    :return: encoded queries
    """
    all_X = []
    all_A = []
    all_E = []
    all_y = []
    time = 0
    for i in range(len(file_names)):
        if 'star' in file_names[i]:
            X_i, A_i, E_i, y_i, time_start_i = read_star_graph_pattern(d, b, n, e, file_names[i], train_tuples, matrix_mode)
        else:
            X_i, A_i, E_i, y_i, time_start_i = read_chain_graph_pattern(d, b, n, e , file_names[i], train_tuples, matrix_mode)
        all_X.append(X_i)
        all_A.append(A_i)
        all_E.append(E_i)
        all_y.append(y_i)
        time += time_start_i
        print("Read " + file_names[i])

    X = np.concatenate(all_X, axis = 0)
    A = np.concatenate(all_A, axis = 0)
    E = np.concatenate(all_E, axis = 0)
    y = np.concatenate(all_y, axis = 0)

    return X, A, E, y, time