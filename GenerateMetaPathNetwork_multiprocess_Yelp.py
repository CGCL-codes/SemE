import networkx as nx
import random
from config import args
import functools
import multiprocessing


def compare(A, B):
    a = int(A[1:])
    b = int(B[1:])
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


def partition(lst, n):
    division = len(lst) / float(n)
    return [list(lst)[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]


class GenerateMetaNetwork:
    """Generate semantic networks based on meta-path and star topology for dataset Yelp
    In Yelp, to share some methods, users, business, and stars are replaced with authors, papers, and conferences

    methods:  generate_graph(), generate_semantic_node(), and generate_semantic_network(), running().
    """
    def __init__(self, Meta):
        """
        :param Meta: the set of meta-paths
        """
        super()
        self.dataset = args.dataset
        self.output_path = args.output_path
        self.G, self.author_list, self.paper_list, self.conf_list = self.generate_graph()
        self.list_dict = {"a": self.author_list, "p": self.paper_list, "c": self.conf_list}
        self.list_len = {"a": len(self.author_list), "p": len(self.paper_list), "c": len(self.conf_list)}
        self.Meta = Meta

    def generate_graph(self):
        """construct the temporal HIN

        :return G: the temporal HIN
        :return author_list: all of users in network
        :return paper_list: all of businesses in network
        :return conf_list: all of stars in network
        """
        G = nx.Graph()
        G_edgelist = nx.Graph()
        author_list = []
        paper_list = []
        conf_list = []

        for i in range(args.author_num + args.paper_num + args.conf_num):
            G_edgelist.add_node(i)
        with open("data/output/" + self.output_path + "/business_user.txt", mode="r", encoding="UTF-8") as f:
            for line in f:
                s = line.strip("\n").split(" ")
                if "p" + s[0] not in paper_list:
                    paper_list.append("p" + s[0])
                if "a" + s[1] not in author_list:
                    author_list.append("a" + s[1])
                G.add_node("p" + s[0], node_type="paper")
                G.add_node("a" + s[1], node_type="author")
                G.add_edge("p" + s[0], "a" + s[1], time=s[2])
                G_edgelist.add_edge(int(s[0]) - 1 + args.author_num, int(s[1]) - 1, weight=1)

        with open("data/output/" + self.output_path + "/business_star.txt", mode="r", encoding="UTF-8") as f:
            for line in f:
                s = line.strip("\n").split(" ")
                if "p" + s[0] not in paper_list:
                    paper_list.append("p" + s[0])
                if "c" + s[1] not in conf_list:
                    conf_list.append("c" + s[1])
                G.add_node("p" + s[0], node_type="paper")
                G.add_node("c" + s[1], node_type="conf")
                G.add_edge("p" + s[0], "c" + s[1], time=s[2])
                G_edgelist.add_edge(int(s[0]) - 1 + args.author_num, int(s[1]) - 1 + args.author_num + args.paper_num, weight=1)

        # save temporal HIN with .edgelist file
        nx.write_edgelist(G_edgelist, path="data/output/" + self.output_path + "/original_graph.edgelist", data=[('weight', int)])

        # sort nodes from small to large
        author_list.sort(key=functools.cmp_to_key(compare))
        paper_list.sort(key=functools.cmp_to_key(compare))
        conf_list.sort(key=functools.cmp_to_key(compare))
        return G, author_list, paper_list, conf_list

    def generate_semantic_node(self, M, node_list):
        """sample semantic nodes from the temporal HIN

        :param M: mate-path
        :param node_list: the node set with the same type as M[0]
        :return: meta: set of semantic nodes
        """
        meta = []
        dict_edge_type = nx.get_edge_attributes(self.G, "time")

        for _ in range(args.sample_times):
            print(str(_) + "th sampling")

            # A complete sampling process
            for l in range(len(node_list)):
                if l % 1000 == 0:
                    print(l, "/", len(node_list))

                start = node_list[l]

                for num in range(args.sample_num):
                    tmp_meta = [start]
                    for i in M[1:]:
                        neighbors = self.G.neighbors(tmp_meta[-1])
                        match_neighbors = []

                        # Delete nodes whose types do not match
                        for neighbor in neighbors:
                            if neighbor[0] == i:
                                match_neighbors.append(neighbor)
                        if len(match_neighbors) == 0:
                            break
                        next = random.choice(match_neighbors)
                        tmp_meta.append(next)

                    if len(tmp_meta) != len(M):
                        continue

                    # discard semantic nodes containing duplicate nodes
                    meaningful = True
                    for i in range(len(tmp_meta) // 2):
                        if tmp_meta[i] == tmp_meta[-1 - i]:
                            meaningful = False
                    if not meaningful:
                        continue

                    # time information of nodes in the semantic node
                    time_list = []
                    t1 = dict_edge_type[tmp_meta[1], tmp_meta[0]] \
                            if (tmp_meta[1], tmp_meta[0]) in list(dict_edge_type.keys()) \
                            else dict_edge_type[tmp_meta[0], tmp_meta[1]]
                    t2 = dict_edge_type[tmp_meta[-1], tmp_meta[-2]] \
                            if (tmp_meta[-1], tmp_meta[-2]) in list(dict_edge_type.keys()) \
                            else dict_edge_type[tmp_meta[-2], tmp_meta[-1]]

                    if M == "pap":
                        time_list = [t1, str((int(t1) + int(t2)) // 2), t2]
                    elif M == "pcp":
                        time_list = [t1, str((int(t1) + int(t2)) // 2), t2]

                    # Avoid adding the same semantic node twice
                    node = "-".join(tmp_meta) + " " + " ".join(time_list)
                    tmp_meta.reverse()
                    time_list.reverse()
                    node_reverse = "-".join(tmp_meta) + " " + " ".join(time_list)

                    bool3 = node in meta
                    bool4 = node_reverse in meta
                    if bool3 or bool4:
                        pass
                    else:
                        meta.append(node)

        return meta

    def generate_semantic_network(self, meta, M):
        """Generate semantic network based on meta-path and star topology

        :param meta: set of semantic nodes
        :param M: meta-path
        """
        G = nx.Graph()
        length = len(meta)
        for m in meta:
            G.add_node(m.split(" ")[0])

        for i in range(args.author_num):
            G.add_node("a" + str(i + 1))

        for i in range(args.paper_num):
            G.add_node("p" + str(i + 1))

        for i in range(args.conf_num):
            G.add_node("c" + str(i + 1))

        for i in range(len(meta)):
            if i % 10000 == 0:
                print(i, "/", len(meta))
            source_node = meta[i].split(" ")[0].split("-")

            for node in source_node:
                if node[0] == "a":
                    G.add_edge(i, int(node[1:]) - 1 + length, weight=1)
                elif node[0] == "p":
                    G.add_edge(i, int(node[1:]) - 1 + length + args.author_num, weight=1)
                elif node[0] == "c":
                    G.add_edge(i, int(node[1:]) - 1 + length + args.author_num + args.paper_num, weight=1)

        nx.write_edgelist(G, path="data/output/" + self.output_path + "/" + M + "_graph.edgelist", data=[('weight', int)])
        with open("data/output/" + self.output_path + "/" + M + "_meta_dict.txt", mode="w", encoding="UTF-8") as f:
            for i in range(length):
                f.write(str(i) + " " + meta[i] + "\n")

    def running(self):
        """ generate semantic nodes by multi-threading and construct semantic networks

        :return:
        """
        pool = multiprocessing.Pool(processes=args.processes_num)

        for i in self.Meta:
            print("Meta:", i)
            result = []
            node_list = partition(self.list_dict[i[0]], args.processes_num)
            for j in range(args.processes_num):
                result.append(pool.apply_async(self.generate_semantic_node, (i, node_list[j])))
            meta = []
            for res in result:
                meta += res.get()
            self.generate_semantic_network(meta=meta, M=i)
        pool.close()
        pool.join()
