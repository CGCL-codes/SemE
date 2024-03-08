from GenerateMetaPathNetwork_multiprocess_Yelp import GenerateMetaNetwork
from config import args
import os
from proNE import prone_func, generate_structure_embedding
from Net_link_pre_Yelp import running
from preprocess import get_dataset_link_pre
import time
import numpy as np

"""
To share some methods, we treat users, businesses and stars as authors, papers and conferences respectively.
For example, ubu is equivalent to apa
"""
Meta = ["apa", "apcpa"]

if __name__ == "__main__":
    start = time.perf_counter()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    print(args)

    t1 = time.perf_counter()
    auc_result = []
    f1_result = []
    acc_result = []
    for i in range(args.runners):
        generateSemanticNetwork = GenerateMetaNetwork(Meta=Meta)
        generateSemanticNetwork.running()

        generate_structure_embedding()
        for m in Meta:
            prone_func(m)
        train_loader, test_loader = get_dataset_link_pre(args.train_size)
        result = running(train_loader, test_loader)
        auc_result.append(result[0])
        f1_result.append(result[1])
        acc_result.append(result[2])
    t2 = time.perf_counter()
    print("time per runner:{}".format((t2 - t1) / args.runners))
    print("auc: ave:{}, std:{}".format(np.mean(auc_result), np.std(auc_result, ddof=1)))
    print("f1: ave:{}, std:{}".format(np.mean(f1_result), np.std(f1_result, ddof=1)))
    print("acc: ave:{}, std:{}".format(np.mean(acc_result), np.std(acc_result, ddof=1)))
    end = time.perf_counter()
    print("total time:{}".format(end - start))

