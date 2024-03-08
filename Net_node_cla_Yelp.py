import torch
import torch.nn.functional as F
from torch import nn
from utils import read_semantic_embedding, get_semantic_network_size, get_edge_index_and_interval, \
    read_structure_embedding
from sklearn.metrics import f1_score, accuracy_score
from config import args
from torch_scatter import scatter

device = torch.device('cuda:' + str(args.GPU_ID) if torch.cuda.is_available() else 'cpu')


class Hawkes_process_Layer(nn.Module):
    """
    semantic evolution modeling
    """

    def __init__(self, output_dim):
        super(Hawkes_process_Layer, self).__init__()
        self.meta_size = output_dim
        self.params = nn.Parameter(torch.zeros(args.dim, 1), requires_grad=True)

    def forward(self, interval, embedding, edge_index):
        """
        neighbor propagation by temporal
        :param interval: the interval matrix between the time of semantic nodes and the current time t
        :param embedding: the embedding of semantic nodes
        :param edge_index: the set of edges which are in the semantic network
        :return: the influence of semantic nodes
        """
        row, col = edge_index
        # the attenuation parameter
        theta = torch.mm(embedding, self.params)
        theta = theta[col]
        time_decay_factor = torch.exp(interval.view(-1, 1) * theta)

        x_j = embedding[col]
        x_j = time_decay_factor * x_j
        embedding = scatter(x_j, row, dim=0)

        return embedding[self.meta_size + args.author_num: self.meta_size + args.author_num + args.paper_num]


class Concat_Layer(nn.Module):
    """
    concat several embedding including semantic embedding and structure embedding
    """

    def __init__(self):
        super(Concat_Layer, self).__init__()
        self.linear = nn.Linear(args.dim * 3, args.dim)

    def forward(self, apa_embedding, apcpa_embedding, original_embedding):
        embedding = torch.cat((apa_embedding, apcpa_embedding, original_embedding), 1)
        embedding = self.linear(embedding)
        embedding = F.relu(embedding)
        return embedding


class Net(nn.Module):
    def __init__(self, output_dim0, output_dim1, output_dim):
        super(Net, self).__init__()
        self.hawkes_process_layer1 = Hawkes_process_Layer(output_dim0)
        self.hawkes_process_layer2 = Hawkes_process_Layer(output_dim1)
        self.concat_Layer = Concat_Layer()

        self.linear1 = nn.Linear(args.dim, args.dim * 5)
        self.linear2 = nn.Linear(args.dim * 5, output_dim)
        self.linear3 = nn.Linear(args.dim, args.dim)
        self.linear4 = nn.Linear(args.dim, args.dim)
        self.linear5 = nn.Linear(args.dim, args.dim)
        self.linear6 = nn.Linear(args.dim, args.dim)
        self.linear7 = nn.Linear(args.dim, args.dim)
        self.linear8 = nn.Linear(args.dim, args.dim)
        self.linear9 = nn.Linear(args.dim, args.dim)
        self.linear10 = nn.Linear(args.dim, args.dim)
        self.linear11 = nn.Linear(args.dim, args.dim)
        self.linear12 = nn.Linear(args.dim, args.dim)

    def forward(self, semantic_embedding0, interval0, edge_index0,
                semantic_embedding1, interval1, edge_index1,
                central_node_emb0, central_node_emb1, data, structure_embedding):
        embedding0 = self.linear3(semantic_embedding0)
        embedding1 = self.linear4(semantic_embedding1)

        embedding0 = self.hawkes_process_layer1(interval0, embedding0, edge_index0)
        embedding0 = F.relu(embedding0)
        embedding1 = self.hawkes_process_layer2(interval1, embedding1, edge_index1)
        embedding1 = F.relu(embedding1)

        embedding0 = self.linear6(embedding0[data])
        embedding1 = self.linear7(embedding1[data])

        emb0 = self.linear9(central_node_emb0)
        emb1 = self.linear10(central_node_emb1)

        embedding0 = emb0 + embedding0
        embedding1 = emb1 + embedding1

        structure_embedding = self.linear12(structure_embedding)
        embedding = self.concat_Layer(embedding0, embedding1, structure_embedding)

        embedding = self.linear1(embedding)
        embedding = F.relu(embedding)
        embedding = F.dropout(embedding, training=self.training, p=args.dropout)
        embedding = self.linear2(embedding)

        return embedding


def running(train_loader, test_loader):
    dataset = args.output_path
    meta_length0 = get_semantic_network_size(dataset + "/pap_meta_dict.txt")
    meta_length1 = get_semantic_network_size(dataset + "/pcp_meta_dict.txt")

    # bub
    edge_index0, interval0 = get_edge_index_and_interval(dataset + "/pap_meta_dict.txt",
                                                         dataset + "/pap_graph.edgelist")
    semantic_embedding0 = read_semantic_embedding(dataset + "/pap-prone-embedding.txt",
                                                  meta_length0,
                                                  meta_length0 + args.author_num + args.paper_num + args.conf_num)

    # bsb
    edge_index1, interval1 = get_edge_index_and_interval(dataset + "/pcp_meta_dict.txt",
                                                         dataset + "/pcp_graph.edgelist")
    semantic_embedding1 = read_semantic_embedding(dataset + "/pcp-prone-embedding.txt",
                                                  meta_length1,
                                                  meta_length1 + args.author_num + args.paper_num + args.conf_num)

    apa_central_node_emb = torch.tensor(semantic_embedding0[-1 * (args.paper_num + args.conf_num):
                                                            -1 * args.conf_num]).float().to(device)
    interval0 = torch.tensor(interval0).float().to(device)
    edge_index0 = torch.LongTensor(edge_index0).to(device)

    apcpa_central_node_emb = torch.tensor(semantic_embedding1[-1 * (args.paper_num + args.conf_num):
                                                              -1 * args.conf_num]).float().to(device)
    interval1 = torch.tensor(interval1).float().to(device)
    edge_index1 = torch.LongTensor(edge_index1).to(device)

    semantic_embedding0 = torch.tensor(semantic_embedding0).float().to(device)
    semantic_embedding1 = torch.tensor(semantic_embedding1).float().to(device)

    structure_embedding = read_structure_embedding(dataset + "/original-prone-embedding.txt",
                                                   args.author_num + args.paper_num + args.conf_num)
    structure_embedding = torch.tensor(
        structure_embedding[args.author_num: args.author_num + args.paper_num]).float().to(device)

    model = Net(meta_length0, meta_length1, args.num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print("------train-------")
    result = [0.0, 0.0]
    for epoch in range(args.epochs):
        y_pred = []
        y_true = []
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(semantic_embedding0=semantic_embedding0, interval0=interval0, edge_index0=edge_index0,
                        semantic_embedding1=semantic_embedding1, interval1=interval1, edge_index1=edge_index1,
                        central_node_emb0=apa_central_node_emb[data], central_node_emb1=apcpa_central_node_emb[data],
                        data=data, structure_embedding=structure_embedding[data])
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            _, y_pred_batch = out.max(dim=1)
            y_pred += y_pred_batch.detach().cpu().tolist()
            y_true += target.detach().cpu().tolist()
            acc = accuracy_score(target.detach().cpu(), y_pred_batch.detach().cpu())
            print("epoch:{}, batch:{}, loss:{}, acc:{}".format(epoch, batch_idx, loss, acc))

        model.eval()
        for batch, (data_test, target_test) in enumerate(test_loader):
            data_test, target_test = data_test.to(device), target_test.to(device)
            _, pred = model(semantic_embedding0=semantic_embedding0, interval0=interval0, edge_index0=edge_index0,
                            semantic_embedding1=semantic_embedding1, interval1=interval1, edge_index1=edge_index1,
                            central_node_emb0=apa_central_node_emb[data_test],
                            central_node_emb1=apcpa_central_node_emb[data_test],
                            data=data_test, structure_embedding=structure_embedding[data_test]).max(dim=1)
            micro_f1 = f1_score(target_test.cpu(), pred.cpu(), average='micro')
            macro_f1 = f1_score(target_test.cpu(), pred.cpu(), average='macro')
            acc_test = accuracy_score(target_test.cpu(), pred.cpu())
            print('Micro_F1_score:{}, Macro_F1_score:{}, acc:{}'.format(micro_f1, macro_f1, acc_test))
            if micro_f1 + macro_f1 > sum(result):
                result[0] = micro_f1
                result[1] = macro_f1
    print("the best result:\nMacro_F1_score:{}, Micro_F1_score:{}".format(result[1], result[0]))
    return result
