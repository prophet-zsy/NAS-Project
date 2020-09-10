import os
import json
import copy
import shutil
from chart_generate import chart_gen
from read_net_info import read_network_info, get_y


def gen_report(time_for_run, date, place, item_path, relative_item_path, setting, add_data_blk):
	
	# title info
	content = "## (日期)"+date+"\n"
	content += "**"+place+",  "+relative_item_path+"**\n"
	content += "### "+os.path.basename(relative_item_path).replace("_OK", "").replace("_ok", "").replace("-OK", "").replace("-ok", "")+"_"+str(time_for_run)+"\n\n"
	content += "#### 参数\n\n"
	content += "**搜索空间：**\n\n"
	
	# space
	search_space = {}
	search_space["block_num"] = setting["nas_main"]["block_num"]
	if "enum_debug" in setting["enum"].keys():
		del setting["enum"]["enum_debug"]
	if "enum_log_path" in setting["enum"].keys():
		del setting["enum"]["enum_log_path"]
	search_space["graph"] = setting["enum"]
	search_space["skip_max_dist"] = setting["spl"]["skip_max_dist"]
	search_space["skip_max_num"] = setting["spl"]["skip_max_num"]
	if "pooling" in setting["spl"]["space"].keys():
		del setting["spl"]["space"]["pooling"]
	search_space["ops"] = setting["spl"]["space"]
	search_space_str = json.dumps(search_space, indent=4)
	# add indent
	search_space_str += "\n"
	search_space_str = "    " + search_space_str
	search_space_str = search_space_str.replace("\n", "\n    ")
	search_space_str += "\n"
	search_space_str = search_space_str.replace("\n                        ", "").replace("\n                    ]", "]")

	content += search_space_str
	
	# config
	content += "**搜索配置：**\n\n"
	config = {}
	config["num_gpu"] = setting["nas_main"]["num_gpu"]
	config["num_opt_best"] = setting["nas_main"]["num_opt_best"]
	config["finetune_threshold"] = setting["nas_main"]["finetune_threshold"]
	config["spl_network_round"] = setting["nas_main"]["spl_network_round"]
	config["eliminate_policy"] = setting["nas_main"]["eliminate_policy"]
	config["add_data_mode"] = setting["nas_main"]["add_data_mode"]
	if "repeat_search" in setting["nas_main"].keys():
		config["repeat_search"] = setting["nas_main"]["repeat_search"]
	elif "repeat_search" in setting["eva"].keys():
		config["repeat_search"] = setting["eva"]["repeat_search"]
	else:
		raise KeyError("can not find the key repeat_search in the nas_config file")
	if config["add_data_mode"] == "linear":
		config["add_data_per_round"] = setting["nas_main"]["add_data_per_round"]
	else:
		config["init_data_size"] = setting["nas_main"]["init_data_size"]
		config["data_increase_scale"] = setting["nas_main"]["data_increase_scale"]
	config["add_data_for_confirm_train"] = setting["nas_main"]["add_data_for_confirm_train"]
	keys_del = []
	for key in setting["eva"].keys():
		if "path" in key:
			keys_del.append(key)
	for key in keys_del:
		del setting["eva"][key]
	if "retrain_switch" in setting["eva"].keys():
		del setting["eva"]["retrain_switch"]
	config["eva"] = setting["eva"]
	config["spl_para"] = {"opt_para":setting["opt"]}
	config_str = json.dumps(config, indent=4)
	# add indent
	config_str = "    " + config_str
	config_str += "\n"
	config_str = config_str.replace("\n", "\n    ")
	config_str += "\n"

	content += config_str

	# net structure
	content += "#### 网络结构\n"

	with open(os.path.join(item_path, 'nas_log.txt'), "r") as f:
		log = f.readlines()
		# find the num of round in every block
		id = 0
		while True:
			if "doing_task" in log[-id]:
				temp = log[-id]
				round_per_blk = temp.split(" ")[2].split(":")[-1]
				break
			else:
				id += 1
		# find the num of network
		id = 0
		while True:
			if "NAS: Now we have " in log[id]:
				temp = log[id]
				net_num = temp.split(" ")[4]
				break
			else:
				id += 1
		# find the retrain score and retrain time
		retrain_info = log[-1].replace("  ", " ")
		retrain_time = retrain_info.split(" ")[-1]
		retrain_score = retrain_info.split(" ")[-4]
		retrain_time = int(retrain_time.split(".")[0])
		retrain_score = float(retrain_score[:6])
		# find the final network structure and search time
		block_num = search_space["block_num"]
		search_time = log[-2-block_num]
		search_time = search_time.split(" ")[-3]
		search_time = int(search_time.split(".")[0])
		graph = log[-1-block_num:-1]
		graph = "".join(graph)
		graph = "[" + graph
		graph = graph[:-1]
		graph += "]"
		graph = eval(graph)

	#  repeat the block
	repeat_search = config["repeat_search"]
	new_graph = []
	for blk in graph:
		for _ in range(repeat_search):
			blk_cp = copy.deepcopy(blk)
			blk_cp[0].append([])
			blk_cp[1].append(('identity'))
			new_graph.append(blk_cp)
		new_graph[-1][1][-1] = ('pooling', 'max', 2)
	graph = new_graph

	graph_str = str(graph)
	graph_str = graph_str.replace("]],", "]],    \n")
	graph_str += "    \n\n"
	content += graph_str

	# net structure pic
	content += "![diagram](./"+os.path.basename(relative_item_path).replace("_OK", "").replace("_ok", "").replace("-OK", "").replace("-ok", "")+"-result_"+str(time_for_run)+".png \"diagram\")\n\n"

	# process chart
	content += "#### 过程统计\n\n"
	content += "每轮信息：\n\n"
	content += "| 进度 | 数据量     | 最高评分    |   网络个数    |   总用时/s     |  每个网络平均用时/s|  每轮每个网络采样次数 |  加速比  |\n"
	content += "|----| --------    | -----:      |   -----     |   ----         |     -----       |  -------- | ----- |\n"
	net_path = os.path.join(item_path, 'network_info.txt')
	spl_times = config["spl_network_round"]
	blk_num = search_space["block_num"]
	round_per_blk = int(round_per_blk)
	num_opt_best = config["num_opt_best"]
	net_num = int(net_num)
	log_path = os.path.join(item_path, 'nas_log.txt')
	if os.path.basename(relative_item_path)[:10] in add_data_blk:  # add data in blk mode 
		data_scale = "linear_block"
	else:
		data_scale = config["add_data_mode"]
	data_per_round = setting["nas_main"]["add_data_per_round"]
	init_data_size = setting["nas_main"]["init_data_size"]
	data_increase_scale = setting["nas_main"]["data_increase_scale"]
	chart = chart_gen(net_path=net_path, spl_times=spl_times, blk_num=blk_num, round_per_blk=round_per_blk,
							num_opt_best=num_opt_best, net_num=net_num, log_path=log_path, data_scale=data_scale, 
							data_per_round=data_per_round, init_data_size=init_data_size, data_increase_scale=data_increase_scale)
	
	content += chart
	content += "\n\n"

	content += "最好评分变化：\n\n"
	headline = "    "
	for i in range(round_per_blk):
		headline += "round "
		headline += str(i+1)
		headline += " "
	for _ in range(config["num_opt_best"]//config["spl_network_round"]-1):
		headline += "round "
		headline += str(round_per_blk)
		headline += " "
	headline += "\n"
	content += headline
	net_info = read_network_info(os.path.join(item_path, 'network_info.txt'))
	_, _, _, _, _, var_score = get_y(net_info, spl_times)
	content += var_score
	content += "\n\n"

	content += "search用时:   \n"
	content += str(search_time)
	content += "s   \n"
	content += "retrain用时:   \n"
	content += str(retrain_time)
	content += "s   \n"
	content += "retrain评分:   \n"
	content += str(retrain_score)
	content += "   \n\n"

	content += "#### 曲线图\n\n"
	content += "![diagram](./"+os.path.basename(relative_item_path).replace("_OK", "").replace("_ok", "").replace("-OK", "").replace("-ok", "")+"_"+str(time_for_run)+".png \"diagram\")\n\n"
	content += "-----------------\n\n\n"

	return content


if __name__ == '__main__':
	report_path = "./exp_report/exp_repo_restruct/report01_13.md"
	add_data_blk = ['NAS-c100-2', 'NAS-c100-3']

	time_for_run = 7
	date = "2019/01/13"
	place = "华为"
	dir_name = 'NAS_0113'
	dir_path = os.path.join(os.getcwd(), dir_name)
	items = os.listdir(dir_path)
	for item in items:
		sub_dir_path = os.path.join(dir_path, item)
		sub_items = os.listdir(sub_dir_path)
		for item in sub_items:
			if ('OK' in item or 'ok' in item) and '-1' not in item:
				print("######processing@@@@@@@", item)
				relative_item_path = os.path.join(dir_name, item)
				if relative_item_path[-3:].isdigit():
					relative_item_path = relative_item_path[:-6]
				item_path = os.path.join(sub_dir_path, item)

				source = os.path.join(item_path, 'best_score', 'score_with_spl.png')
				target = os.path.join(os.getcwd(), 'exp_report', 'exp_repo_restruct', os.path.basename(relative_item_path).replace("_OK", "").replace("_ok", "").replace("-OK", "").replace("-ok", "")+"_"+str(time_for_run)+".png")
				shutil.copyfile(source, target)
				source = os.path.join(item_path, os.path.basename(relative_item_path).replace("_OK", "").replace("_ok", "").replace("-OK", "").replace("-ok", "")+"-result("+str(time_for_run)+').png')
				target = os.path.join(os.getcwd(), 'exp_report', 'exp_repo_restruct', os.path.basename(relative_item_path).replace("_OK", "").replace("_ok", "").replace("-OK", "").replace("-ok", "")+"-result_"+str(time_for_run)+'.png')
				shutil.copyfile(source, target)

				with open(os.path.join(item_path, 'nas_config.json')) as f:
					setting = json.load(f)
				report_content = gen_report(time_for_run, date, place, item_path, relative_item_path, setting, add_data_blk)
				# print(report_content)
				with open(report_path, "a", encoding='utf-8') as f:
					f.write(report_content)
