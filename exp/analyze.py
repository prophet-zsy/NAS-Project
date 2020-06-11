import pickle, os, sys, copy, json
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import base
from functools import cmp_to_key
from gen_gv_file import blk_to_png
from graph_subtraction import subtraction
import cv2


class Analyzer:
    def __init__(self):
        self._cur_dir = os.getcwd()
        self._bin_dir = os.path.join(self._cur_dir, 'exp', 'bin')
        if not os.path.exists(self._bin_dir):
            os.mkdir(self._bin_dir)
        self.memory_path =  os.path.join(self._cur_dir, "memory/")
        self.base_data_dir = os.path.join(self.memory_path, "base_data_serialize")
        self.stage_path = os.path.join(self.memory_path, "stage_info.pickle")

        self.net_pool = []
        self._load_base_data()

        # self.stage_info = None
        # self._load_stage()

        self.blk_num = -1
        self.nn_num = -1
        self.spl_batch = -1
        self._init_param()

        self._check_data()

    def _load_stage(self):
        with open(self.stage_path, "r") as f:
            self.stage_info = json.load(f)

    def _load_base_data(self):  # ordered list
        # sort the path list
        paths = os.listdir(self.base_data_dir)
        if not paths:
            raise Exception("There is nothing in the folder base_data_serialize")
        def get_blk_id(item):
            return int(item.split("_")[1])
        def get_item_id(item):
            return int(item.split("_")[-1].split(".")[0])
        def cmp(a, b):
            a_blk_id = get_blk_id(a)
            b_blk_id = get_blk_id(b)
            if a_blk_id == b_blk_id:
                return get_item_id(a) - get_item_id(b)
            return a_blk_id - b_blk_id
        paths.sort(key=cmp_to_key(cmp))
        # init the net_pool
        for item in paths:
            dump_path = os.path.join(self.base_data_dir, item)
            with open(dump_path, "rb") as f_dump:
                net = pickle.load(f_dump)
                self.net_pool.append(net)

    def _check_data(self):
        for nn_id in range(self.nn_num):
            tem_graph_part = self.net_pool[nn_id].graph_template
            for blk_id in range(1, self.blk_num):
                if self.net_pool[blk_id*self.nn_num+nn_id].graph_template != tem_graph_part:
                    raise Exception("graph_part of every block is not same, you must check it by your self")
    
    def _init_param(self):
        self.blk_num = len(self.net_pool[-1].item_list[-1].task_info.pre_block)+1
        self.nn_num = len(self.net_pool)//self.blk_num
        self.spl_batch = self.net_pool[0].item_list[0].task_info.spl_batch_num

    def get_graph_parts(self):
        #  graph_part scheme of all the block are the same
        #  return the original graph_template
        graph_templates = [copy.deepcopy(nn.graph_template) for nn in self.net_pool[:self.nn_num]]
        return graph_templates

    def get_items(self, blk_num, round):
        i = blk_num  # - 1
        j = round # - 1
        items = []
        for nn in self.net_pool[i*self.nn_num:blk_num*self.nn_num]:
            items.extend(nn.item_list[j*self.spl_batch:round*self.spl_batch])
            #  for last round, in train winner num_opt_best > spl_batch_num
            if round*self.spl_batch<len(nn.item_list) and\
                nn.item_list[round*self.spl_batch].task_info.round == round:
                items.extend(nn.item_list[round*self.spl_batch:])
        return items

    def get_item(self, blk_id, nn_id, item_id):
        item = self.net_pool[blk_id*self.nn_num+nn_id].item_list[item_id]
        return item

    def get_item_use_pred(self, blk_id, nn_id):
        nn = self.net_pool[blk_id*self.nn_num+nn_id]
        items = [item for item in nn.item_list if item.use_pred]
        assert len(items) == 1, "there are many items in one nn using pred,"\
                                "please make sure you use not only one pred in every nn"
        item = items[0]
        return item.id, item

    def reappear_search(self, blk_id="all"):
        search_process = []
        max_item_num = 0
        for i in range(self.blk_num):
            blk_search = []
            for j in range(i*self.nn_num, (i+1)*self.nn_num):
                scores = [item.score for item in self.net_pool[j].item_list]
                blk_search.append(scores)
                if len(scores) > max_item_num:
                    max_item_num = len(scores)
            search_process.append(blk_search)
        
        for blk_search in search_process:
            for nn in blk_search:
                while len(nn) < max_item_num:
                    nn.append("-")
        
        # display
        blk_ids = [blk_id] if blk_id is not "all" else [idx for idx in range(self.blk_num)]
        for i in blk_ids:
            print("blk_id: {}/{}".format(i, self.blk_num))
            blk_search = search_process[i]
            print("\t"+"\t".join([str(i) for i in range(len(blk_search[0]))]))
            for nn_id in range(len(blk_search)):
                print(nn_id, end="")
                cnt = 0
                for score in blk_search[nn_id]:
                    if cnt < self.spl_batch and self.net_pool[i*self.nn_num+nn_id].item_list[cnt].use_pred:  # the item use pred
                        if score >= max(blk_search[nn_id][:self.spl_batch]):
                            aux_info = " pd_ok"
                        else:
                            aux_info = " pd"
                    else:
                        aux_info = ""
                    context = "\t{:.3f}{}".format(score, aux_info) if score != "-" else\
                              "\t{}".format(score)
                    print(context, end="")
                    cnt += 1
                print()
            print("\n")
        return search_process

        # graph_parts = self.get_graph_parts()
        # #TODO display graph_parts
        # for i in range(1,1+self.blk_num):
        #     rd = 1
        #     while True:
        #         items = self.get_items(i, rd)
        #         #TODO display items for cur round
        #         if not items:
        #             break
        #         rd += 1
        #     #TODO display confirm train
        # #TODO display search result

    def show_png(self, png_path):
        png_path = png_path + ".png"
        img = cv2.imread(png_path)
        png_name = os.path.basename(png_path)
        cv2.imshow(png_name, img)
        cv2.waitKey(2000)

    def get_graph_part(self, nn_id):
        graph_part = copy.deepcopy(self.net_pool[nn_id].graph_template)
        identity_id = len(graph_part)
        old_tail = graph_part.index([])
        graph_part[old_tail] = [identity_id]
        graph_part.append([])
        return graph_part

    def display_graph_part(self, nn_id):
        graph_part = self.get_graph_part(nn_id)
        cell_list = ['None' for _ in range(len(graph_part)-1)]
        cell_list.append('identity')

        blk_graph = [[graph_part, cell_list]]
        graph_part_name = 'graph_part_nn{}'.format(nn_id)
        output_path = os.path.join(self._bin_dir, graph_part_name)
        blk_to_png(blk_graph, output_path)
        # self.show_png(output_path)

    def _item_to_blkgraph(self, item, with_preblk=False):
        #  get the pre_blk and concat them with cur item, then return them to func "display_item" to draw them
        task_info = item.task_info
        blk_list = task_info.pre_block + [item]
        blk_graph = []
        mark_skipping = []
        for item in blk_list:
            cell_list = [tuple(item) for item in item.cell_list]
            graph, cell_list = copy.deepcopy(item.graph), copy.deepcopy(cell_list)
            graph.append([])
            cell_list.append(('identity'))
            blk_graph.append([graph, cell_list])

            nn_id = item.task_info.nn_id
            graph_part = self.get_graph_part(nn_id)
            # print(graph, graph_part)
            skipping = subtraction(graph, graph_part)
            mark_skipping.append([skipping])
        return blk_graph, mark_skipping

    def display_item(self, blk_id, nn_id, item_id, with_preblk=False):
        item = self.get_item(blk_id, nn_id, item_id)
        task_info = item.task_info
        # you can display following content
        # item.graph, item.cell_list, item.code, item.score, 
        # task_info.graph_template, task_info.pre_block, task_info.round
        # task_info.cost_time
        blk_graph, mark_skipping = self._item_to_blkgraph(item, with_preblk=with_preblk)

        item_name = 'blk{}_nn{}_item{}'.format(blk_id, nn_id, item_id)
        output_path = os.path.join(self._bin_dir, item_name)
        blk_to_png(blk_graph, output_path, mark_skip_edge=mark_skipping)
        # self.show_png(output_path)

    def display_item_use_pred(self, blk_id, nn_id):
        nn = self.net_pool[blk_id*self.nn_num+nn_id]
        for item_id in range(len(nn.item_list)):
            item = nn.item_list[item_id]
            if item.use_pred:
                self.display_item(blk_id, nn_id, item_id, with_preblk=True)
                return blk_id, nn_id, item_id
        raise Exception("We assert that there is at least one item used pred, but we found none!")

    def display_items_first_round(self, blk_id, nn_id):
        nn = self.net_pool[blk_id*self.nn_num+nn_id]
        first_round_items = nn.item_list[:self.spl_batch]
        for idx, item in enumerate(first_round_items):
            self.display_item(blk_id, nn_id, idx, with_preblk=True)
            if item.use_pred:
                print(idx, "use_pred")
            else:
                print(idx)
            print(item.graph)
            print(item.cell_list)
            print(item.code)
            print(item.score)

    def compute_pred_work_rate(self):
        work_cnt = 0
        for nn in self.net_pool:
            first_round_items = nn.item_list[:self.spl_batch]
            max_score = 0
            score_use_pred = 0
            for item in first_round_items:  # get the max_score and score_use_pred
                max_score = item.score if item.score > max_score else max_score
                if item.use_pred:
                    score_use_pred = item.score
            if score_use_pred == max_score:  # pred work in this nn
                work_cnt += 1
        return work_cnt/len(self.net_pool)
    
    # def get_search_result(self):


        
if __name__ == "__main__":
    analyze = Analyzer()
    # print(analyze.nn_num)
    # print(analyze.get_graph_parts())
    # print(analyze.get_items(1, 1))
    analyze.reappear_search(blk_id="all")
    analyze.display_item(blk_id=3, nn_id=5, item_id=0, with_preblk=True)
    # for i in range(analyze.nn_num):
    #     analyze.display_graph_part(nn_id=i)
    # print(analyze.get_item_use_pred(blk_id=2, nn_id=1))
    # analyze.display_items_first_round(blk_id=0, nn_id=0)
    # print(analyze.compute_pred_work_rate())
    # 如何查看单个枚举的网络，从头到尾的变化
        

    
