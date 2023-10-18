import numpy as np
from data_processor_release import numpy_binary

class QueryGraph:
    """ Query Graph: containing the query graph for learning """
    def __init__(self, d, b, n, e):
        """
        Initialization of the query graph variables
        :param d: int, the number of distinct nodes (subjects + objects) in KG
        :param b: int, the number of distinct edges (predicates) in KG
        :param n: int, the number of nodes in the subgraph
        :param e: int, the number of edges in the subgraph
        """
        self.nodes = set()
        self.edges = set()
        self.all_triples = []
        self.cardinality = -1
        self.d = d
        self.b = b
        self.n = n
        self.e = e

    def add_triple(self, triple, mapping=None):
        """
        Adding a triple to the query graph
        :param triple: of form s, p, o
        :return:
        """
        self.all_triples.append(triple)
        self.nodes.add(triple[0])
        self.nodes.add(triple[2])
        self.edges.add(triple[1])

    def print(self):
        print("Nodes " + str(self.nodes) + ", Edges " + str(self.edges) + ", Triples " + str(self.all_triples))

    def create_graph(self):
        """
        Creation of the query graph
        """
        self.nodes_sorted = list(self.nodes)
        self.nodes_sorted.sort()
        self.edges_sorted = list(self.edges)
        self.edges_sorted.sort()
        # print("Sorted nodes: " + str(self.nodes_sorted) + ", Sorted edges: " + str(self.edges_sorted))


        ''' Initialization of the required matrices '''
        # matrix_mode == 3:
        bits_d = int(np.ceil(np.log2(self.d))) + 1
        bits_b = int(np.ceil(np.log2(self.b))) + 1
        # print("Bits d %d" % bits_d + ", Bits b %d" % bits_b)
        x = np.zeros((self.n, bits_d), dtype='uint8')
        ep = np.zeros((self.e, bits_b), dtype='uint8')
        a = np.zeros((self.e, self.n, self.n), dtype='uint8')
        # print("Shape of x is " + str(np.shape(x)) + ", Shape of ep is " + str(np.shape(ep)))

        ''' Setting a '''
        for triple in self.all_triples:
            s, p, o = triple
            s_idx = self.nodes_sorted.index(s)
            o_idx = self.nodes_sorted.index(o)
            p_idx = self.edges_sorted.index(p)
            # print("s: " + str(s_idx) + ", " + "p: " + str(p_idx) + ", " + "o: " + str(o_idx))
            a[p_idx, s_idx, o_idx] = 1
        # print("Shape of x is " + str(np.shape(x)) + ", Shape of ep is " + str(np.shape(ep)))

        ''' Setting ep '''
        for p_idx, predicate in enumerate(self.edges_sorted):
            e_number = self.handle_variable(predicate)
            arr = numpy_binary(e_number, bits_b)
            # print("The predicate " + str(predicate) + ", in binary " + str(arr))
            for i in range(len(arr)):
                ep[p_idx][i] = arr[i]

        ''' Setting x '''
        for n_idx, node in enumerate(self.nodes_sorted):
            n_number = self.handle_variable(node)
            arr = numpy_binary(n_number, bits_d)
            # print("The node " + str(node) + ", in binary " + str(arr))
            for i in range(len(arr)):
                x[n_idx][i] = arr[i]

        return x, ep, a, self.cardinality


    def handle_variable(self, x):
        """
        Handling a term
        :param x: Term is either a bound term-number or an unbound term (variable)-star
        :return: int number for bound term, 0 for unbound term
        """
        if "*" in x or "?" in x:
            return 0
        return int(x)