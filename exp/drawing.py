import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import json


cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}



def read_network_info(name):
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
    for block in y_list:
        for net in block:
            print(net)

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
    print(y_sample)

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
    for block in y_list_best:
        for net in block:
            print(net)

    y_list_multi_spl = copy.deepcopy(y_list_best)  #  record all the best score with sequence of the spl
    y_sample_best = []
    for block in range(len(y_list_multi_spl)):
        while y_list_multi_spl[block] != []:
            for network in range(len(y_list_multi_spl[block])):
                # print(len(y_list_multi_spl[block]))
                # print(len(y_list_multi_spl[block][network]))
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
    print(y_sample_best)

    return y_list, y_list_best, y_sample, y_sample_best


def show(path, keep_multi=1, spl_times=6, x_length=10, save_path=None):
    net_path = os.path.join(path, 'network_info.txt')
    network_info = read_network_info(net_path)
    x = [i+1 for i in range(x_length*spl_times)] if keep_multi else [i+1 for i in range(x_length)]
    y_list, y_list_best, y_sample, y_sample_best = get_y(network_info, spl_times)
    y = y_list if keep_multi else y_list_best  # determined whether keep the multi spl

    for block in range(len(y)):
        best_y = [0 for _ in range(x_length*spl_times)] if keep_multi else [0 for _ in range(x_length)]
        colors = cnames.keys()
        color_list = []
        for color in colors:
            color_list.append(color)
        plt.title('Block'+str(block))
        print('Block'+str(block))
        for network in range(len(y[block])):
            network_store = y[block][network]
            print(network_store)
            print(len(network_store))
            for id in range(len(network_store)):
                if id < len(best_y) and network_store[id] > best_y[id]:  # here, we remove the more 2 score in winner
                    best_y[id] = network_store[id]
            x_true_length = x_length*spl_times if keep_multi else x_length
            while len(network_store) < x_true_length:
                network_store.append(0)
            while len(network_store) > x_true_length:
                network_store.pop(-1)
            plt.plot(x, network_store, '-', color=color_list[network%len(color_list)], label=str(network))

        plt.legend()  # 显示图例

        plt.xlabel('round')
        plt.ylabel('score')
        if save_path:
            pic_path = os.path.join(save_path, str(block)+'all.png')
            plt.savefig(pic_path)
            plt.close()
        else:
            plt.show()
        

        plt.title('best_Block'+str(block))
        plt.plot(x, best_y, '-', color="red", label="best_score_current_round")
        plt.legend()  # 显示图例

        plt.xlabel('round')
        plt.ylabel('score')
        if save_path:
            pic_path = os.path.join(save_path, str(block)+'best.png')
            plt.savefig(pic_path)
            plt.close()
        else:
            plt.show()
        

    y_sample = y_sample_best if not keep_multi else y_sample
    # y_sample = [item for item in y_sample if item > 0.12]  # remove the 0.1 score
    x_sample = [i+1 for i in range(len(y_sample))]

    # from matplotlib.backends.backend_pdf import PdfPages
    # plt.figure(figsize=(8, 6))
    # plt.clf()

    # plt.plot(x_sample, y_sample, '-', color="red", label="eva_score_with_spl_times")

    # plt.xlabel('sample_num', fontsize=19)
    # plt.ylabel('score', fontsize=19)

    # # plt.xticks(fontsize=15)
    # # plt.yticks(fontsize=15)
    # # plt.grid(False)
    # # plt.legend(fontsize=13, loc='upper left')
    # plt.ylim(0.4, 1)
    # pdf_file = 'relevance.pdf'
    # pp = PdfPages(pdf_file)
    # pp.savefig()
    # pp.close()

    # plt.show()



    plt.plot(x_sample, y_sample, '-', color="red", label="eva_score_with_spl_times")
    plt.xlabel('sample_num')
    plt.ylabel('score')
    if save_path:
        pic_path = os.path.join(save_path, 'score_with_spl.png')
        plt.savefig(pic_path)
        plt.close()
    else:
        plt.show()
    


if __name__ == '__main__':
    dir_path = os.path.join(os.getcwd(), 'NAS_0113')
    items = os.listdir(dir_path)
    for item in items:
        sub_dir_path = os.path.join(dir_path, item)
        sub_items = os.listdir(sub_dir_path)
        for item in sub_items:
            if 'OK' in item or 'ok' in item:
                print("######processing@@@@@@@", item)
                item_path = os.path.join(sub_dir_path, item)
                with open(os.path.join(item_path, 'nas_config.json')) as f:
                    setting = json.load(f)
                    spl_times = setting["nas_main"]["spl_network_round"]
                best_score_path = os.path.join(item_path, 'best_score')
                keep_multi_path = os.path.join(item_path, 'keep_multi')

                if not os.path.exists(best_score_path):
                    os.makedirs(best_score_path)
                if not os.path.exists(keep_multi_path):
                    os.makedirs(keep_multi_path)

                show(item_path, keep_multi=1, spl_times=spl_times, x_length=10, save_path=keep_multi_path)
                show(item_path, keep_multi=0, spl_times=spl_times, x_length=10, save_path=best_score_path)


