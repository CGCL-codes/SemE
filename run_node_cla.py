from GenerateMetaPathNetwork_multiprocess import GenerateMetaNetwork
from config import args
import os
from proNE import prone_func, generate_structure_embedding
from Net_node_cla import running
from preprocess import get_dataset_node_classify
import time
import numpy as np


Meta = ["apa", "apcpa", "appa"]

if __name__ == "__main__":
    start = time.perf_counter()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    print(args)

    t1 = time.perf_counter()

    micro_result = []
    macro_result = []
    for i in range(args.runners):
        generateSemanticNetwork = GenerateMetaNetwork(Meta=Meta)
        generateSemanticNetwork.running()

        generate_structure_embedding()
        for m in Meta:
            prone_func(m)

        train_loader, test_loader = get_dataset_node_classify(args.train_size)
        result = running(train_loader, test_loader)
        micro_result.append(result[0])
        macro_result.append(result[1])

    t2 = time.perf_counter()
    print("time per runner:{}".format((t2 - t1) / args.runners))
    print("train size: {}%".format(args.train_size * 100))
    print("micro_f1: ave:{}, std:{}".format(np.mean(micro_result), np.std(micro_result, ddof=1)))
    print("macro_f1: ave:{}, std:{}".format(np.mean(macro_result), np.std(macro_result, ddof=1)))
    end = time.perf_counter()
    print("total time:{}".format(end - start))


