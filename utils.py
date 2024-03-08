import numpy as np
from config import args


def get_semantic_network_size(meta_file_path):
    """
    get the number of semantic nodes
    :param meta_file_path: the file which saves semantic nodes
    :return: count: the number of semantic nodes
    """
    count = 0
    with open(meta_file_path, mode="r", encoding="UTF-8") as f:
        for line in f:
            count += 1
    return count


def read_semantic_embedding(embedding_file_path, meta_num, all_num):
    """
    read the semantic embedding
    :param embedding_file_path: the file which saves semantic embedding
    :param meta_num: the number of semantic nodes
    :param all_num: the number of nodes which are in the semantic network
    :return: embedding: the embedding of semantic nodes and central nodes
    """
    embeddings = np.zeros((all_num, args.dim))
    node_num = 0
    node_list = []
    with open(embedding_file_path, mode="r", encoding="UTF-8") as f:
        for line in f:
            if len(line) < args.dim:
                continue
            s = line.strip("\n").split(" ")
            embedding = s[1:]
            temp = map(eval, embedding)
            embeddings[int(s[0])] = list(temp)
            if int(s[0]) >= meta_num:
                node_num += 1
                node_list.append(int(s[0]))

    all_embeddings = np.sum(embeddings[meta_num: all_num], axis=0)
    for i in range(meta_num, all_num):
        if i not in node_list:
            embeddings[i] = all_embeddings / node_num

    return embeddings


def read_structure_embedding(embedding_file_path, all_num):
    """
    read structure embedding
    :param embedding_file_path: the file which saves structure embedding
    :param all_num: the number of nodes which are in the temporal HIN
    :return:
    """
    embeddings = np.zeros((all_num, args.dim))
    with open(embedding_file_path, mode="r", encoding="UTF-8") as f:
        for line in f:
            if len(line) < args.dim:
                continue
            s = line.strip("\n").split(" ")
            embedding = s[1:]
            temp = map(eval, embedding)
            embeddings[int(s[0])] = list(temp)

    return embeddings


def get_edge_index_and_interval(meta_file_path, edge_file_path):
    """
    :param meta_file_path: the file which saves semantic nodes
    :param edge_file_path: the .edgelist of the semantic network
    :return: edge_index: the set of edges which are in the semantic network
             interval: the set of interval between the time of semantic nodes and the current time t
    """
    id_meta_dict = {}
    with open(meta_file_path, mode="r", encoding="UTF-8") as f:
        for line in f:
            s = line.strip("\n").split(" ")
            id_meta_dict[s[0]] = line[len(s[0]):].strip()

    meta_num = len(id_meta_dict)

    edge_interval = []
    edge_index = [[], []]
    with open(edge_file_path, mode="r", encoding="UTF-8") as f:
        for line in f:
            s = line.strip("\n").split(" ")
            if int(s[0]) >= meta_num and int(s[1]) >= meta_num:
                continue

            if int(s[0]) >= meta_num:
                edge_index[0].append(int(s[0]))
                edge_index[1].append(int(s[1]))
                time = id_meta_dict[s[1]].split(" ")
                time = list(map(eval, time[1:]))
                edge_interval.append(max(time) - args.latest_time)

            else:
                edge_index[0].append(int(s[1]))
                edge_index[1].append(int(s[0]))
                time = id_meta_dict[s[0]].split(" ")
                time = list(map(eval, time[1:]))
                edge_interval.append(max(time) - args.latest_time)

    return edge_index, edge_interval
