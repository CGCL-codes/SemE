import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import random
from config import args

label_dict = {"pos": 1, "neg": 0}


class MyDataset_NL(torch.utils.data.Dataset):
    """
    dataset class for node classification
    """
    def __init__(self, data, transform=None, target_transform=None):
        super(MyDataset_NL, self).__init__()
        self.data = data
        self.transform = transform
        self.target_tansform = target_transform

    def __getitem__(self, item):
        x, y = self.data[item]

        x = torch.tensor(x)

        return x, y

    def __len__(self):
        return len(self.data)


class MyDataset_LP(torch.utils.data.Dataset):
    """
    dataset class for link prediction
    """
    def __init__(self, data, transform=None, target_transform=None):
        super(MyDataset_LP, self).__init__()
        self.data = data
        self.transform = transform
        self.target_tansform = target_transform

    def __getitem__(self, item):
        (x0, x1), y = self.data[item]

        x0 = torch.tensor(x0)
        x1 = torch.tensor(x1)

        return x0, x1, y

    def __len__(self):
        return len(self.data)


def read_file(path, label):
    data = []
    with open(path, mode="r", encoding="UTF-8") as f:
        for line in f:
            s = line.strip("\n").split(" ")
            data.append([(int(s[0]) - 1, int(s[1]) - 1), label_dict[label]])
    return data


def read_classify_data(path):
    x = []
    y = []
    data = []
    with open(path, mode="r", encoding="UTF-8") as f:
        for line in f:
            s = line.strip("\n").split(" ")
            x.append(int(s[0]) - 1)
            y.append(int(s[1]))
            data.append([int(s[0]) - 1, int(s[1])])
    return data


def get_dataset_node_classify(train_size):
    dataset = args.dataset
    head = "/author"
    if dataset == "Yelp_node_cla":
        head = "/business"

    data = read_classify_data(dataset + head + "_label.txt")
    random.shuffle(data)
    train_dataset = data[:int(train_size * len(data))]
    test_dataset = data[int(train_size * len(data)):]
    print("generate mini-batch")
    train_data = MyDataset_NL(train_dataset, transform=transforms.ToTensor())
    test_data = MyDataset_NL(test_dataset, transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_dataset))
    print("data prepare done")
    return train_loader, test_loader


def get_dataset_link_pre(train_size):
    dataset = args.dataset
    head = "/Coauthor"
    if dataset == "Yelp_link_pre":
        head = "/couser"
    pos_data = read_file("data/input/" + dataset + head + "_Pos.txt", "pos")
    neg_data = read_file("data/input/" + dataset + head + "_Neg.txt", "neg")

    train_dataset = pos_data[:int(train_size * len(pos_data))] + neg_data[:int(train_size * len(neg_data))]
    test_dataset = pos_data[int(train_size * len(pos_data)):] + neg_data[int(train_size * len(neg_data)):]

    print(len(train_dataset))
    random.shuffle(train_dataset)
    print("generate mini-batch")
    train_data = MyDataset_LP(train_dataset, transform=transforms.ToTensor())
    test_data = MyDataset_LP(test_dataset, transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=len(test_dataset))
    print("data prepare done")

    return train_loader, test_loader
