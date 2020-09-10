import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy


def read_network_info(name):
    """
    Args:
    name: must ends with "network.txt"
    """
    f = open(name, 'r')
    lines = f.readlines()
    network_info = []
    block_num = -1
    # round_num = -1
    block = []
    network = []
    for line in lines:
        if line[:11] == "block_num: ":  # a new network
            if block_num == int(line[11]):  # still in current block
                if network:
                    block.append(network)
                # round_num = int(line[20])
            else:  # a new block
                block_num = int(line[11])
                if block:
                    block.append(network)
                    network_info.append(block)
                block = []
            network = []
            continue
        if line[:11] == "graph_part:":
            network.append(line)
            continue
        if line[:15] == "    graph_full:":
            network.append([line])
            continue
        if line[:14] == "    cell_list:":
            network[-1].append(line)
            continue
        if line[:10] == "    score:":
            network[-1].append(float(line[10:]))
            continue
    block.append(network)
    network_info.append(block)

    return network_info


def get_y(network_info, spl_times):
    y_list = []
    for block in network_info:
        blk = []
        for network in block:
            y = []
            for content in network:
                if len(content) == 3:
                    y.append(content[-1])
                    # print(content[-1])
            blk.append(y)
        y_list.append(blk)
    # for block in y_list:
    #     for net in block:
    #         print(net)

    y_list_multi_spl = copy.deepcopy(y_list)  #  record all the score with sequence of the spl
    y_sample = []
    for block in range(len(y_list_multi_spl)):
        while y_list_multi_spl[block] != []:
            for network in range(len(y_list_multi_spl[block])):
                # print(len(y_list_multi_spl[block]))
                # print(len(y_list_multi_spl[block][network]))
                y_sample.extend(y_list_multi_spl[block][network][:spl_times])
                del y_list_multi_spl[block][network][:spl_times]
                # print("rest score",len(y_list_multi_spl[block][network]))
            delete = []
            for network in range(len(y_list_multi_spl[block])):
                if y_list_multi_spl[block][network] == []:
                    delete.append(network)
            add = 0
            for num in delete:
                del y_list_multi_spl[block][num+add]
                add -= 1
    # print(y_sample)

    y_list_best = copy.deepcopy(y_list)
    for block in y_list_best:  #  get best score from multi spl
        for network in range(len(block)):
            new_y = []
            best_score = 0
            for part in range(len(block[network])):
                if block[network][part] > best_score:
                    best_score = block[network][part]
                if (part+1)%spl_times == 0:
                    new_y.append(best_score)
                    best_score = 0
            block[network] = new_y
    tmp_str = ""
    for block in y_list_best:
        for net in block:
            # print(net)
            tmp = [eval('{:.4f}'.format(i)) for i in net]
            tmp_str += "    "
            tmp_str += str(tmp)
            tmp_str += "\n"
            # print(tmp)


    y_list_multi_spl = copy.deepcopy(y_list_best)  #  record all the best score with sequence of the spl
    y_sample_best = []
    for block in range(len(y_list_multi_spl)):
        while y_list_multi_spl[block] != []:
            for network in range(len(y_list_multi_spl[block])):
                # print(y_list_multi_spl[block])
                # print(y_list_multi_spl[block][network])
                y_sample_best.append(y_list_multi_spl[block][network][0])
                del y_list_multi_spl[block][network][0]
                # print("rest score",len(y_list_multi_spl[block][network]))
            delete = []
            for network in range(len(y_list_multi_spl[block])):
                if y_list_multi_spl[block][network] == []:
                    delete.append(network)
            add = 0
            for num in delete:
                del y_list_multi_spl[block][num+add]
                add -= 1
    # print(y_sample_best)

    y_list_average = copy.deepcopy(y_list)
    for block in y_list_average:  #  compute average score from multi spl
        for network in range(len(block)):
            new_y = []
            sum_score = 0
            for part in range(len(block[network])):
                if block[network][part] > best_score:
                    sum_score += block[network][part]
                if (part+1)%spl_times == 0:
                    new_y.append(sum_score/spl_times)
                    sum_score = 0
            block[network] = new_y
    # tmp_str = ""
    # for block in y_list_average:
    #     for net in block:
    #         # print(net)
    #         tmp = [eval('{:.4f}'.format(i)) for i in net]
    #         tmp_str += str(tmp)
    #         tmp_str += "\n"
    #         # print(tmp)

    return y_list, y_list_best, y_sample, y_sample_best, y_list_average, tmp_str


if __name__ == '__main__':
    net_path = 'D:/personal_zhangshuyu/my_document/论文/NAS/CompetitionNAS/Experiment/NAS_12_18/6-cifar100_ok/network_info.txt'
    spl_times = 6
    info = read_network_info(net_path)
    y_list, y_list_best, y_sample, y_sample_best, y_list_average, tmp_str = get_y(info, spl_times)
    # for block in y_list_best:
    #     round_best = [0 for _ in range(10)]
    #     id_best = [-1 for _ in range(10)]
    #     # print(block)
    #     for net_id in range(len(block)):
    #         # print(net)
    #         for k in range(len(block[net_id])):
    #             # print("####", block[net_id][k])
    #             # print("****", round_best[k])
    #             if block[net_id][k] > round_best[k]:
    #                 round_best[k] = block[net_id][k]
    #                 id_best[k] = net_id+1
    #     print(round_best)
    #     print(id_best)
