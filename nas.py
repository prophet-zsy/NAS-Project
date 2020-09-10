import time, random, os, sys, re, copy
import numpy as np
import traceback
from base import Network, NetworkItem
from utils import TaskScheduler, EvaScheduleItem, PredScheduleItem
from utils import DataSize, _epoch_ctrl, NAS_LOG, Logger, TimeCnt
from utils import _dump_stage, _check_log
from info_str import NAS_CONFIG, Stage_Info

from enumerater import Enumerater
from sampler import Sampler
task_name = NAS_CONFIG['eva']['task_name']
if task_name == "denoise":
    from evaluator_denoise import Evaluator
elif task_name == "SIDD":
    from evaluator_denoise_SIDD import Evaluator
else:
    from evaluator_classification import Evaluator

MAIN_CONFIG = NAS_CONFIG['nas_main']
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

TSche = TaskScheduler()


def _subproc_eva(task_item, result_buffer, signal, eva):
    NAS_LOG = Logger()
    task_item.pid = os.getpid()
    time_cnt = TimeCnt()
    task_item.start_time = time_cnt.start()
    if task_item.network_item:
        NAS_LOG << ('nas_eva_ing', len(task_item.pre_block)+1,\
            task_item.round, task_item.nn_id, task_item.network_item.id)
    if MAIN_CONFIG['eva_mask']:
        task_item.score = random.uniform(0, 0.1)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(task_item.gpu_info)
        try:
            task_item.score = eva.evaluate(task_item)
        except Exception as error:
            _err_log(NAS_LOG, task_item, error)
            task_item.score = 0
            
    task_item.cost_time = time_cnt.stop()
    if task_item.network_item:
        NAS_LOG << ('nas_eva_fin', len(task_item.pre_block)+1,\
                task_item.round, task_item.nn_id, task_item.network_item.id,\
                task_item.score, task_item.model_params, task_item.cost_time, task_item.pid)

    #  use in subprocess
    if result_buffer and signal:
        result_buffer.put(task_item)
        signal.set()
        
    return task_item

def _err_log(NAS_LOG, task_item, error):
    pre_block = []
    for block in task_item.pre_block:
        pre_block.append((block.graph, block.cell_list, block.code))
    NAS_LOG << ('err_task_info', TimeCnt().start(), len(pre_block)+1, task_item.nn_id,\
                task_item.alig_id, str(pre_block), str(task_item.graph_template))
    scheme = task_item.network_item
    NAS_LOG << ('err_scheme_info', task_item.task_id, task_item.pid, task_item.start_time, \
    task_item.cost_time, task_item.gpu_info, task_item.round, task_item.nn_left, \
        task_item.spl_batch_num, str(scheme.graph), str(scheme.cell_list), \
            str(scheme.code), scheme.score)
    NAS_LOG << ('err_info', error)

def _save_net_info(net_rm, round, net_left):
    net_serialize = "basedata"
    net_info_temp = "net_info-elim_net_info"
    sche_info_temp = "net_info-scheme_info"
    NAS_LOG << (net_serialize, net_rm)
    blk_num, rd, net_lft, net_id, sche_num, graph_temp = \
        len(net_rm.pre_block)+1, round, net_left, net_rm.id, len(net_rm.item_list), net_rm.graph_template
    NAS_LOG << (net_info_temp, blk_num, rd, net_lft, net_id, sche_num, graph_temp)
    for scheme in net_rm.item_list:
        task_info = scheme.task_info
        NAS_LOG << (sche_info_temp, task_info.task_id, task_info.pid, task_info.start_time, \
            task_info.cost_time, task_info.gpu_info, task_info.round, task_info.nn_left, \
                task_info.spl_batch_num, str(scheme.graph), str(scheme.cell_list), \
                    str(scheme.code), scheme.score)


def _pred_ops(nn, pred, graph, table):
    pre_block = [elem.graph.copy() for elem in Network.pre_block]
    for block in pre_block:
        if block[-1]:
            block.append([])
    graph.append([])  # add the virtual node
    pred_ops = pred.predictor(pre_block, graph)
    pred_ops = pred_ops[:-1]  # remove the ops of virtual node
    table = nn.spl.ops2table(pred_ops, table)
    cell, graph = nn.spl.convert(table)
    return graph, cell, table

def _sample_batch(network, batch_num=1, pred=None):
    """sample with duplicate check
    """
    graphs, cells, tables = [], [], []
    use_pred = []
    spl_index = 0
    cnt = 0
    while spl_index < batch_num:
        cnt += 1
        if cnt > 500:
            NAS_LOG << ('nas_no_dim_spl', spl_index)
            raise ValueError("sample error")
        cell, graph, table = network.spl.sample()
        if pred:
            graph, cell, table = _pred_ops(network, pred, graph, table)
        if table not in tables:
            graphs.append(graph)
            cells.append(cell)
            tables.append(table)
            spl_index += 1
            if pred:
                use_pred = table  # record the table which used pred
                pred = None  # only pred one item in the init
    
    start_id = len(network.item_list)
    item_ids = range(start_id, start_id + batch_num)
    for item_id, graph, cell, table in zip(item_ids, graphs, cells, tables):
        network.item_list.append(NetworkItem(item_id, graph, cell, table, use_pred==table))
    return item_ids

def _sample(net_pool, batch_num=MAIN_CONFIG['spl_network_round'], base_alig_id=[], pred=None):
    if base_alig_id:
        base_item_id = []
        for idx in base_alig_id:
            newly_item_id = _sample_batch(net_pool[idx], batch_num, pred)
            for item_id in newly_item_id:
                base_item_id.append((idx, item_id))
        return base_item_id
    else:
        for nn in net_pool:
            _sample_batch(nn, batch_num, pred)

def _update_batch(network, batch_num=1):
    for idx in range(1, batch_num+1):
        network.spl.update_opt_model(network.item_list[-idx].code, -network.item_list[-idx].score)


def _update(net_pool, batch_num=1, newly_added_id=[]):
    if newly_added_id:
        for idx in newly_added_id:
            net_update = net_pool[idx[0]]
            net_update.spl.update_opt_model(net_update.item_list[idx[1]].code, -net_update.item_list[idx[1]].score)
    else:
        for network in net_pool:
            _update_batch(network, batch_num=batch_num)


def _spl_info_to_tasks(net_pool, round, cur_epoch, cur_data_size, batch_num=MAIN_CONFIG['spl_network_round'], base_item_id=[]):
    pool_len = len(net_pool)
    finetune_sign = (pool_len < MAIN_CONFIG['finetune_threshold'])
    isbestNN = False
    task_list = []
    if base_item_id:
        for idx in base_item_id:
            nn = net_pool[idx[0]]
            pre_blk = nn.pre_block
            nn_id = nn.id
            graph_template = nn.graph_template
            item = nn.item_list[idx[1]]
            task_item = EvaScheduleItem(nn_id, idx[0], graph_template,\
                                        item, pre_blk, finetune_sign,\
                                        isbestNN, round, pool_len, batch_num,\
                                        cur_epoch, cur_data_size)
            task_list.append(task_item)
    else:
        for alig_id, nn in enumerate(net_pool):
            pre_blk = nn.pre_block
            nn_id = nn.id
            graph_template = nn.graph_template
            for item in nn.item_list[-batch_num:]:
                task_item = EvaScheduleItem(nn_id, alig_id, graph_template,\
                                            item, pre_blk, finetune_sign,\
                                            isbestNN, round, pool_len, batch_num,\
                                            cur_epoch, cur_data_size)
                task_list.append(task_item)
    return task_list

def _eva_net(task_list, eva, async_exec=False):
    if MAIN_CONFIG['subp_eva_debug']:
        result = []
        for task_item in task_list:
            task_item = _subproc_eva(task_item, None, None, eva)
            result.append(task_item)
    else:
        TSche.load_tasks(task_list)
        if async_exec:
            TSche.exec_task_async(_subproc_eva, eva)
            TSche.load_part_result()
        else:
            TSche.exec_task(_subproc_eva, eva)
        result = TSche.get_result()
    return result

def _record_result(net_pool, result):
    newly_added_id = []
    for task_item in result:
        item = net_pool[task_item.alig_id].item_list[task_item.network_item.id]
        item.score = task_item.score
        item.task_info = task_item
        newly_added_id.append((task_item.alig_id, task_item.network_item.id))
    return newly_added_id

def _choose_policy(batch_num=MAIN_CONFIG['spl_network_round']):
    obj, pattern = MAIN_CONFIG['eliminate_policy'].split("_")
    if obj == "history":
        idx = 0
    elif obj == "cur":
        idx = - batch_num
    if pattern == "best":
        policy = max
    elif pattern == "average":
        policy = np.mean
    return idx, policy

def _eliminate(net_pool, round, batch_num=MAIN_CONFIG['spl_network_round']):
    idx, policy = _choose_policy(batch_num)
    scores = [policy([x.score for x in net_pool[nn_id].item_list[idx:]])
              for nn_id in range(len(net_pool))]
    scores = [item for item in list(enumerate(scores))]
    original_len = len(scores)
    scores.sort(key=lambda x: x[1])
    scores_rm = scores[:original_len//2]
    scores_rm.sort(key=lambda x: x[0], reverse=True)
    pre_blk = net_pool[0].pre_block
    for item in scores_rm:
        net_rm = net_pool.pop(item[0])
        NAS_LOG << ("nas_elim_net", len(pre_blk) + 1, round, len(net_pool),
                    net_rm.id, len(net_rm.item_list))
        _save_net_info(net_rm, round, len(net_pool))
    NAS_LOG << ('nas_eliinfo_tem', len(scores_rm), len(scores)-len(scores_rm))
    
def _game(eva, net_pool, ds, round):
    time_cnt = TimeCnt()
    start_round = time_cnt.start()
    block_id = len(net_pool[0].pre_block)
    NAS_LOG << ('nas_round_start', block_id+1, round, start_round)
    cur_data_size = ds.control(stage="game")
    cur_epoch = _epoch_ctrl(eva, stage="game")
    if round > 1:
        round_template = copy.deepcopy(Stage_Info['blk_info'][block_id]['round_info'][0])
        Stage_Info['blk_info'][block_id]['round_info'].append(round_template)
    Stage_Info['blk_info'][block_id]['round_info'][-1]['round_start'] = start_round
    Stage_Info['blk_info'][block_id]['round_info'][-1]['round_data_size'] = cur_data_size
    Stage_Info['blk_info'][block_id]['search_epoch'] = cur_epoch

    if round > 1:
        _sample(net_pool)
    task_list = _spl_info_to_tasks(net_pool, round, cur_epoch, cur_data_size)
    result = _eva_net(task_list, eva)
    _record_result(net_pool, result)
    _eliminate(net_pool, round)
    _update(net_pool)

    end_round = time_cnt.stop()
    NAS_LOG << ('nas_round_over', end_round)
    Stage_Info['blk_info'][block_id]['round_info'][-1]['round_cost'] = end_round


def _confirm_train(eva, best_nn, best_index, ds):
    time_cnt = TimeCnt()
    start_confirm = time_cnt.start()
    pre_blk = best_nn.pre_block
    blk_id = len(pre_blk)
    NAS_LOG << ("nas_confirm_train", blk_id+1, start_confirm)
    cur_data_size = ds.control(stage="confirm")
    cur_epoch = _epoch_ctrl(eva, stage="confirm")
    Stage_Info['blk_info'][blk_id]['confirm_train_start'] = start_confirm
    Stage_Info['blk_info'][blk_id]['confirm_epoch'] = cur_epoch
    Stage_Info['blk_info'][blk_id]['confirm_data_size'] = cur_data_size

    nn_id = best_nn.id
    alig_id = 0
    graph_template = best_nn.graph_template
    item = best_nn.item_list[best_index]
    network_item = NetworkItem(len(best_nn.item_list), item.graph, item.cell_list, item.code)
    task_list = [EvaScheduleItem(nn_id, alig_id, graph_template, network_item,\
                 pre_blk, ft_sign=True, bestNN=True, rd=-1, nn_left=0, spl_batch_num=1,\
                epoch=cur_epoch, data_size=cur_data_size)]
    if MAIN_CONFIG['subp_eva_debug']:
        result = []
        for task_item in task_list:
            task_item = _subproc_eva(task_item, None, None, eva)
            result.append(task_item)
    else:
        TSche.load_tasks(task_list)
        TSche.exec_task(_subproc_eva, eva)
        result = TSche.get_result()
    network_item.score = result[0].score
    network_item.task_info = result[0]
    best_nn.item_list.append(network_item)

    end_confirm = time_cnt.stop()
    NAS_LOG << ("nas_confirm_train_fin", end_confirm)
    Stage_Info['blk_info'][blk_id]['confirm_trian_cost'] = end_confirm
    return network_item


def _train_winner(eva, net_pl, ds, round, spl_num=MAIN_CONFIG['num_opt_best']):
    """

    Args:
        net_pool: list of NetworkUnit, and its length equals to 1
        round: the round number of game
    Returns:
        best_nn: object of Class NetworkUnit
    """
    time_cnt = TimeCnt()
    start_train_winner = time_cnt.start()
    blk_id = len(net_pl[0].pre_block)
    NAS_LOG << ("nas_train_winner_start", blk_id+1, round, start_train_winner)
    cur_data_size = ds.control(stage="game")
    cur_epoch = _epoch_ctrl(eva, stage="game")
    Stage_Info["blk_info"][blk_id]["train_winner_start"] = start_train_winner
    Stage_Info["blk_info"][blk_id]["train_winner_data_size"] = cur_data_size

    i = 0
    initial = True
    while i < spl_num:
        if initial:
            batch_num = MAIN_CONFIG['num_gpu']
            _sample(net_pl, batch_num=batch_num)
            task_list = _spl_info_to_tasks(net_pl, round, cur_epoch, cur_data_size, batch_num=batch_num)
            result = _eva_net(task_list, eva, async_exec=True)
            newly_added_id = _record_result(net_pl, result)
            initial = False
            i += batch_num
        else:
            newly_num = len(newly_added_id)
            newly_num = newly_num if i+newly_num<spl_num else spl_num-i
            _update(net_pl, newly_added_id=newly_added_id)
            base_alig_id = [idx[0] for idx in newly_added_id]
            base_alig_id = base_alig_id[:newly_num]
            base_item_id = _sample(net_pl, batch_num=1, base_alig_id=base_alig_id)
            async_exec = True if i+newly_num<spl_num else False
            task_list = _spl_info_to_tasks(net_pl, round, cur_epoch, cur_data_size, base_item_id=base_item_id)
            result = _eva_net(task_list, eva, async_exec=async_exec)
            newly_added_id = _record_result(net_pl, result)
            i += newly_num

    best_nn = net_pl.pop(0)
    scores = [x.score for x in best_nn.item_list]
    best_index = scores.index(max(scores)) - len(scores)
    network_item = _confirm_train(eva, best_nn, best_index, ds)
    _save_net_info(best_nn, round, len(net_pl))

    trian_winner_end = time_cnt.stop()
    NAS_LOG << ("nas_train_winner_tem", trian_winner_end)
    Stage_Info["blk_info"][blk_id]["train_winner_cost"] = trian_winner_end
    return network_item

def _round_cmp_topo(net_pool, cmp_couple, blk_id, TopologyEval):
    # add the pre_blk
    pre_blk_graph_part = []
    for item in net_pool[0].pre_block:
        pre_blk_graph_part.append(item.task_info.graph_template)
    
    graph_parts1, graph_parts2 = [], []
    for couple in cmp_couple:
        graph_part1 = pre_blk_graph_part + [net_pool[couple[0]].graph_template]
        graph_part2 = pre_blk_graph_part + [net_pool[couple[1]].graph_template]
        graph_parts1.append(graph_part1)
        graph_parts2.append(graph_part2)
    
    # eva topo1 and topo2, which is better
    res = TopologyEval().topo1vstopo2(graph_parts1, graph_parts2, block_id=blk_id+1)

    # construct dict for find
    cmp_res = {}
    for couple, res_item in zip(cmp_couple, res):
        cmp_res[couple] = res_item
    return cmp_res

def _rm_net(net_pool, cmp_res, rm_num, pred_cnt, pred_consistent_cnt):
    # check consistency and prepare for next rm round
    next_net_pool = []
    cnt = 0
    for nn_id in range(0, len(net_pool), 2):
        pred_cnt += 1
        res1 = cmp_res[(nn_id, nn_id+1)]  # 1 represent that nn_id max
        res2 = cmp_res[(nn_id+1, nn_id)]  # 1 represent that nn_id+1 max
        if res1 + res2 == 1:  # res is consistent
            pred_consistent_cnt += 1
            next_net_pool.append(net_pool[nn_id+res2])
            net_rm = net_pool[nn_id+res1]
            _save_net_info(net_rm, 0, MAIN_CONFIG["max_player_nums"])
            cnt += 1
            if rm_num != -1 and cnt >= rm_num:  # rm cnt achive MAIN_CONFIG["max_player_nums"]
                next_net_pool.extend(net_pool[nn_id+2:])
                break
        else:  # res is not consistent, we keep both of them into next round
            next_net_pool.append(net_pool[nn_id])
            next_net_pool.append(net_pool[nn_id+1])
    return next_net_pool, pred_cnt, pred_consistent_cnt

def _prevent_deadlock(stop_cnt, original_num, net_pool, NAS_LOG):
    if original_num - len(net_pool) == 0:
        stop_cnt += 1
        NAS_LOG << ("nas_filter_topo_stop_cnt", stop_cnt)
        if stop_cnt >= 30:  # if we can not rm net by priori, get first 32 directly
            net_pool = net_pool[:MAIN_CONFIG["max_player_nums"]]
            NAS_LOG << ("nas_filter_topo_get_first_part", MAIN_CONFIG["max_player_nums"])
    return stop_cnt, net_pool

def _filter_topo(net_pool, blk_id):
    if MAIN_CONFIG["filter_mask"]:
        return net_pool
    NAS_LOG = Logger()
    if len(net_pool) <= MAIN_CONFIG["max_player_nums"]:
        NAS_LOG << ("nas_not_filter_topo", len(net_pool), MAIN_CONFIG["max_player_nums"])
        return net_pool

    import keras
    keras.backend.clear_session()
    from TopologyEval import TopologyEval

    pred_consistent_cnt = 0
    pred_cnt = 0
    stop_cnt = 0
    while len(net_pool) > MAIN_CONFIG["max_player_nums"]:
        original_num = len(net_pool)

        rm_num = -1  # when in the last rm round
        if len(net_pool) / 2 < MAIN_CONFIG["max_player_nums"]:
            rm_num = len(net_pool) - MAIN_CONFIG["max_player_nums"]
        
        random.shuffle(net_pool)
        cmp_couple = []
        # add the last net to next round directly if num is odd
        tail_net = None
        if len(net_pool) % 2 == 1:
            tail_net = net_pool[-1]
            net_pool = net_pool[:-1]

        for nn_id in range(0, len(net_pool), 2):
            cmp_couple.append((nn_id, nn_id+1))
            cmp_couple.append((nn_id+1, nn_id))

        cmp_res = _round_cmp_topo(net_pool, cmp_couple, blk_id, TopologyEval)

        net_pool, pred_cnt, pred_consistent_cnt = _rm_net(net_pool, cmp_res, rm_num, pred_cnt, pred_consistent_cnt)
        if tail_net:
            net_pool.append(tail_net)
        
        NAS_LOG << ("nas_filter_net", original_num - len(net_pool), len(net_pool))
        stop_cnt, net_pool = _prevent_deadlock(stop_cnt, original_num, net_pool, NAS_LOG)
        
    assert len(net_pool) == MAIN_CONFIG["max_player_nums"], "nums of nets should be {}, but now it is {}"\
        .format(MAIN_CONFIG["max_player_nums"], len(net_pool))
    NAS_LOG << ("nas_evatopo_consistency_ratio", pred_consistent_cnt/pred_cnt)

    return net_pool

def _init_ops(net_pool):
    import keras
    keras.backend.clear_session()
    from predictor import Predictor
    
    pred = Predictor() if not MAIN_CONFIG["pred_mask"] else None
    _sample(net_pool, batch_num=MAIN_CONFIG['spl_network_round'], pred=pred)
    return net_pool

def _subproc_use_priori(task_item, result_buffer, signal):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(task_item.gpu_info)
    task_item.pid = os.getpid()

    #  get part of the topo
    task_item.net_pool = _filter_topo(task_item.net_pool, task_item.blk_id)

    #  pred ops for every topo
    task_item.net_pool = _init_ops(task_item.net_pool)

    # use in subprocess
    if result_buffer and signal:
        result_buffer.put(task_item)
        signal.set()

    return task_item.net_pool

def _filter_topo_and_init_ops(net_pool, blk_id):
    """Generates ops and skipping for every Network,

    Args:
        net_pool (list of NetworkUnit)
    Returns:
        net_pool (list of NetworkUnit)
        scores (list of score, and its length equals to that of net_pool)
    """
    config_time = TimeCnt()
    config_start = config_time.start()
    NAS_LOG << ('nas_config_ing', config_start)
    Stage_Info['blk_info'][blk_id]['configure_by_priori_start'] = config_start

    task_item = PredScheduleItem(net_pool, blk_id)
    if MAIN_CONFIG['subp_pred_filter_debug']:
        net_pool = _subproc_use_priori(task_item, None, None)
    else:
        TSche.load_tasks([task_item])
        TSche.exec_task(_subproc_use_priori)
        net_pool = TSche.get_result()[0].net_pool

    config_end = config_time.stop()
    NAS_LOG << ('nas_config_fin', config_end)
    Stage_Info['blk_info'][blk_id]['configure_by_priori_cost'] = config_end
    return net_pool

def _init_npool_sampler(netpool, block_id):
    for nw in netpool:
        nw.spl = Sampler(nw.graph_template, block_id)
    return

def _search_blk(block_id, eva, ds, npool_tem):
    """evaluate all the networks asynchronously inside one round and synchronously between rounds
    :param block_id:
    :param eva:
    :param npool_tem:
    :return:
    """
    time_blk = TimeCnt()
    start_block = time_blk.start()
    NAS_LOG << ('nas_search_blk', block_id+1, MAIN_CONFIG["block_num"], start_block)
    Stage_Info['blk_info'][block_id]['blk_start'] = start_block
    
    net_pool = copy.deepcopy(npool_tem)
    _init_npool_sampler(net_pool, block_id)
    # if there are only one net, we do not use the priori to process the net, 
    # and enter the retrain stagess
    if len(net_pool) > 1:
        net_pool = _filter_topo_and_init_ops(net_pool, block_id)
    
    round = 0
    time_game = TimeCnt()
    start_game = time_game.start()
    NAS_LOG << ('nas_rounds_game_start', block_id+1, start_game)
    Stage_Info['blk_info'][block_id]['rounds_game_start'] = start_game

    while len(net_pool) > 1:
        round += 1
        _game(eva, net_pool, ds, round)
    
    game_end = time_game.stop()
    NAS_LOG << ('nas_get_winner', game_end)
    Stage_Info['blk_info'][block_id]['rounds_game_cost'] = game_end
    Stage_Info['blk_info'][block_id]['round_num'] = round

    network_item = _train_winner(eva, net_pool, ds, round + 1)

    blk_end = time_blk.stop()
    NAS_LOG << ('nas_search_blk_end', blk_end)
    Stage_Info['blk_info'][block_id]['blk_cost'] = blk_end
    return network_item

def _retrain(eva, ds):
    time_cnt = TimeCnt()
    start_time = time_cnt.start()
    NAS_LOG << ('nas_retrain', start_time)
    cur_epoch = _epoch_ctrl(eva, stage="retrain")
    cur_data_size = ds.control(stage="retrain")
    task_item = EvaScheduleItem(nn_id=-1, alig_id=-1, graph_template=[], item=None,\
                pre_blk=Network.pre_block, ft_sign=True, bestNN=True, rd=0, nn_left=-1,\
                spl_batch_num=-1, epoch=cur_epoch, data_size=cur_data_size)
    task_list = [task_item]
    TSche.load_tasks(task_list)
    TSche.exec_task(_subproc_eva, eva)
    result = TSche.get_result()
    retrain_score = result[0].score
    retrain_end = time_cnt.stop()
    NAS_LOG << ('nas_retrain_end', retrain_end, retrain_score)
    Stage_Info['retrain_start'] = start_time
    Stage_Info['retrain_cost'] = retrain_end
    Stage_Info['retrain_epoch'] = cur_epoch
    Stage_Info['retrain_data_size'] = cur_data_size

class Nas:
    def __init__(self):
        _check_log()
        NAS_LOG << "nas_init_ing"
        self.enu = Enumerater()
        self.eva = Evaluator() if not MAIN_CONFIG['eva_mask'] else None
        self.ds = DataSize(self.eva)
        
    def run(self):
        NAS_LOG << 'nas_enuming'
        network_pool_tem = self.enu.enumerate()
        NAS_LOG << ('nas_enum_nums', len(network_pool_tem), min(len(network_pool_tem), MAIN_CONFIG['max_player_nums']))
        time_search = TimeCnt()
        start_search = time_search.start()
        NAS_LOG << ('nas_start_search', start_search)
        Stage_Info['nas_start'] = start_search
        for i in range(MAIN_CONFIG["block_num"]):
            network_item = _search_blk(i, self.eva, self.ds, network_pool_tem)
            Network.pre_block.append(network_item)
        end_search = time_search.stop()
        NAS_LOG << ('nas_search_end', end_search)
        Stage_Info['nas_cost'] = end_search
        _dump_stage(Stage_Info)
        for block in Network.pre_block:
            NAS_LOG << ('nas_pre_block', str(block.graph), str(block.cell_list))
        if MAIN_CONFIG['retrain_switch']:
            _retrain(self.eva, self.ds)
        os.system("tar -zcvf nas_log.tar.gz nas_config.json memory")
        os.system("tar -zcvf model.tar.gz model")
        return Network.pre_block


if __name__ == '__main__':
    nas = Nas()
    search_result = nas.run()
    # when interrupt by the error, we can resume from interruption
    # if "resume_inter" in sys.argv:
    #     nas = Nas()
    #     search_result = nas.run()
    # else:
    #     nas = Nas()
    #     search_result = nas.run()

