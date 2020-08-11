import os
import json
import collections

# Current file name
_cur_ver_dir = os.getcwd()

# NAS configuration dic object
_nas_config_path = os.path.join(_cur_ver_dir, 'nas_config.json')

NAS_CONFIG = json.load(open(_nas_config_path, encoding='utf-8'), object_pairs_hook=collections.OrderedDict)

# System information
eva_ing = 'NAS: eva ing -> block_num:{0[0]} round:{0[1]} nn_id:{0[2]} item_id:{0[3]}'
eva_fin = 'NAS: eva fin -> block_num:{0[0]} round:{0[1]} nn_id:{0[2]} item_id:{0[3]}'\
          ' score:{0[4]:.4f} model_params:{0[5]} time_cost:{0[6]} eva_pid:{0[7]}'
eliinfo_tem = 'NAS: eliminating {0[0]}, remaining {0[1]}...'
init_ing = 'NAS: Initializing...'
enum_ing = 'NAS: Enumerating all possible networks!'
enum_toponet_nums = "NAS: We got {0[0]} topo networks, and we will evaluate {0[1]} of them in every block!"
filter_net = "NAS: We rm {0[0]} topo networks by priori, and {0[1]} networks left!"
filter_topo_stop_cnt = "NAS: We can not rm any net int this round, count {0[0]}"
filter_topo_get_first_part = "NAS: We have been not able to rm any net for many rounds, so we get first {0[0]} networks for game!"
evatopo_consist_ratio = "NAS: The ratio that cmp(a,b) and cmp(b,a) is consistency is {0[0]}"
start_search = "NAS: search work start, start_time: {0[0]}"
search_block_start = 'NAS: Searching for block {0[0]}/{0[1]}, start_time: {0[2]}'
not_filter_topo = "NAS: We got {0[0]} networks from enumerater, it is no more than {0[1]}, we will evaluate all of them"
config_ing = 'NAS: We filter the topo net and configure the ops in the first round, start_time: {0[0]}'
config_fin = 'NAS: configure finished and cost_time: {0[0]}'
get_winner = 'NAS: We got a WINNER and cost time: {0[0]}'
best_and_score_tem = 'NAS: We have got the best network and its score is {0[0]:.4f}'
start_game_tem = 'NAS: Now we have {0[0]} networks. Start game!'
train_winner_start = "NAS: train winner start, blk_num: {0[0]} round: {0[1]} start_time: {0[2]}"
search_fin_tem = "NAS: Search finished, cost time: {0[0]} search result:"
pre_block = "[{0[0]}, {0[1]}],"
blk_search_tem = "NAS: Search current block finished and cost time: {0[0]}"
train_winner_tem = "NAS: Train winner finished and cost time: {0[0]}"
rounds_game_start = "NAS: rounds_game start, blk_num: {0[0]} start_time: {0[1]}"
round_start = "NAS: The round start, block_num: {0[0]} round: {0[1]} start_time: {0[2]}"
round_over = "NAS: The round is over, cost time: {0[0]}"
no_dim_spl = "There maybe no dim for sample, {0[0]} table sampled !!!"
elim_net = "Network info of net removed\nblock_num: {0[0]} round: {0[1]} network_left: {0[2]} " \
                "network_id: {0[3]} number of scheme: {0[4]}"
elim_net_info = "block_num: {0[0]} round: {0[1]} network_left: {0[2]} " \
                "network_id: {0[3]} number of scheme: {0[4]}\ngraph_part:{0[5]}\n"
scheme_info = "    task_info:(task_id:{0[0]} pid:{0[1]} start_time:{0[2]} cost_time:{0[3]} gpu_info:{0[4]} round:{0[5]}"\
              " nn_left:{0[6]} spl_batch_num:{0[7]})\n"\
              "    graph_full:{0[8]}\n    cell_list:{0[9]}\n    code:{0[10]}\n    score:{0[11]:.4f}\n"
confirm_train = "NAS: confirm train satrt, blk_num: {0[0]} start_time: {0[1]}"
confirm_train_fin = "confirm train finished and cost time: {0[0]}"
retrain = "NAS: We reconstruct the network which has been searched and evaluate it again, start_time: {0[0]}"
retrain_end = "NAS: Retrain is over, cost time: {0[0]}, score: {0[1]:.4f}"
err_task_info = "EVA Error:(err_time: {0[0]})\n block_num: {0[1]} nn_id: {0[2]} alig_id: {0[3]}\n pre_block: {0[4]}\n graph_part: {0[5]}\n"
err_info = "EVA Error:\n {0[0]} "
eva = "{0[0]}"
log_hint = "There are existed log files in the folder 'memory'.\n"\
       "Please remove and store them if they are useful for you.\n"\
       "And we will cover them by new log, continue?(y/n)y"
existed_log = "Existed log files, we can not rewrite them!"
invalid_str = "Invalid string, please input again:(y/n)"
# ACTION -> logger template string
# in ACTION, the content before "_" represent that we output the string into which file
MF_TEMP = {
    ## nas : you should make the key start with "nas_"
    # run
    'nas_enuming': enum_ing,
    'nas_enum_nums': enum_toponet_nums,
    "nas_filter_net": filter_net,
    "nas_filter_topo_stop_cnt": filter_topo_stop_cnt,
    "nas_filter_topo_get_first_part": filter_topo_get_first_part,
    'nas_evatopo_consistency_ratio': evatopo_consist_ratio,
    'nas_start_search': start_search,
    'nas_search_blk': search_block_start,
    'nas_search_blk_end': blk_search_tem,
    'nas_search_end': search_fin_tem,
    'nas_no_dim_spl': no_dim_spl,
    'nas_pre_block': pre_block,
    # _subproc_eva
    'nas_eva_ing': eva_ing,
    'nas_eva_fin': eva_fin,
    # _algo
    'nas_start_game': start_game_tem,
    'nas_rounds_game_start': rounds_game_start,
    'nas_round_start': round_start,
    'nas_round_over': round_over,
    'nas_get_winner': get_winner,
    # filter topo and pred ops
    'nas_config_ing': config_ing,
    'nas_config_fin': config_fin,
    "nas_not_filter_topo": not_filter_topo,
    # _eliminate
    'nas_eliinfo_tem': eliinfo_tem,
    'nas_elim_net': elim_net,
    # _train_winner
    'nas_train_winner_start': train_winner_start,
    'nas_train_winner_tem': train_winner_tem,
    # __init__
    'nas_init_ing': init_ing,
    # _confirm_train
    'nas_confirm_train': confirm_train,
    'nas_confirm_train_fin': confirm_train_fin,
    # _retrain
    'nas_retrain': retrain,
    'nas_retrain_end': retrain_end,
    # _save_net_info : you should make the key start with "net_"
    'net_info-elim_net_info': elim_net_info,
    'net_info-scheme_info': scheme_info,
    # err log : you should make the key start with "err_"
    'err_task_info': err_task_info,
    'err_scheme_info': scheme_info,
    'err_info': err_info,

    ## evaluator : you should make the key start with "eva_"
    'eva_eva': eva,

    ## enumerater

    ## predictor

    ## sampler

    ## utils
    # hint existed log
    'nas_log_hint': log_hint,
    'nas_existed_log': existed_log,
    'nas_invalid_str': invalid_str,
    # for testing utils
    'enuming': "utils: enuming",
    'hello': "uitls: {0[0]}, {0[1]}"
}

Stage_Info = {
    "nas_start": None,
    "nas_cost": None,
    "blk_info": [
        {
            "blk_start": None,
            "blk_cost": None,
            
            "configure_by_priori_start": None,
            "configure_by_priori_cost": None,
            
            "rounds_game_start": None,
            "search_epoch": None,
            "rounds_game_cost": None,
            "round_num": None,
            "round_info": [
                {
                    "round_start": None,
                    "round_cost": None,
                    "round_data_size": None
                }
            ],

            "train_winner_start": None,
            "train_winner_data_size": None,
            "train_winner_cost": None,

            "confirm_train_start": None,
            "confirm_epoch": None,
            "confirm_data_size": None,
            "confirm_trian_cost": None,
        }
    ],
    "retrain_start": None,
    "retrain_cost": None,
    "retrain_epoch": None,
    "retrain_data_size": None
}
# we can init the blk_info at the begin, but we have to init the round_info in running time
blk_template = Stage_Info['blk_info'][0]
blk_num = NAS_CONFIG['nas_main']['block_num']
Stage_Info['blk_info'] = [blk_template for _ in range(blk_num)]
