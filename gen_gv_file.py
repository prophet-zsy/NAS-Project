import os
import copy
import json
color_list_template = ['yellow', 'green', 'blue', 'orange', 'red', 'cyan', 'pink', 'gold', 'purple', 'firebrick', 
				'olive', 'tomato', 'slategray', 'teal', 'black', 'sienna', 'silver', 'skyblue']


def gen_gv_png(input_path, output_path, output_file_name, block_num, repeat_search, num_classes):
	with open(input_path, "r") as f:
		content = f.readlines()
		graph = content[-1-block_num:-1]
		graph = "".join(graph)
		graph = "[" + graph
		graph = graph[:-1]
		graph += "]"
		graph = eval(graph)

	#  repeat the block
	new_graph = []
	for blk in graph:
		for _ in range(repeat_search):
			blk_cp = copy.deepcopy(blk)
			blk_cp[0].append([])
			blk_cp[1].append(('identity'))
			new_graph.append(blk_cp)
		new_graph[-1][1][-1] = ('pooling', 'max', 2)
	graph = new_graph

	print(graph)

	code_for_write = "digraph G {\n"
	code_for_write += "    0[style=solid,color={},shape=box,label=\"input\"];\n\n".format(color_list.pop(0))
	for blk_id in range(len(graph)):
		code_for_write += "    subgraph cluster_{} ".format(blk_id)
		code_for_write += "{\n"
		code_for_write += "    color=gray;\n"
		code_for_write += "    node [style=solid,color={},shape=box];\n".format(color_list.pop(0))
		for cell_id in range(len(graph[blk_id][1])):
			code_for_write += "    {}{}[label=\"{}\"];\n".format(blk_id+1, cell_id, graph[blk_id][1][cell_id])
		code_for_write += "    label = \"Block{}\";\n".format(blk_id+1)
		code_for_write += "    }\n\n"
	last_color = color_list.pop(0)
	code_for_write += "    1[style=solid,color={},shape=box,label=\"('fc', 256, 'relu')\"];\n".format(last_color)
	code_for_write += "    2[style=solid,color={},shape=box,label=\"('fc', {}, None)\"];\n\n".format(last_color, num_classes)

	code_for_write += "    0 -> 10\n\n"
	for blk_id in range(len(graph)):
		for node_id in range(len(graph[blk_id][0])):
			code_for_write += "    {}{} -> ".format(blk_id+1, node_id)
			if graph[blk_id][0][node_id]:  # if there is any node in the list
				for out_node in graph[blk_id][0][node_id]:
					code_for_write += "{}{},".format(blk_id+1, out_node)
				code_for_write = code_for_write[:-1]
			else:
				if blk_id == len(graph)-1:  # if the block is the last block
					code_for_write += "1"
				else:
					code_for_write += "{}{}".format(blk_id+2, 0)
			code_for_write += "\n"
		code_for_write += "\n"
	code_for_write += "    1 -> 2\n"
	code_for_write += "}\n"

	print(code_for_write)

	with open(output_path, "w") as f:
		f.write(code_for_write)

	output_file_png = output_path.replace(".gv", ".png")

	os.system("dot -Tpng -o "+output_file_png+" "+str(output_path))


if __name__ == '__main__':
	times_for_run = 5

	dir_path = os.path.join(os.getcwd(), 'NAS_0106')
	items = os.listdir(dir_path)
	for item in items:
		sub_dir_path = os.path.join(dir_path, item)
		sub_items = os.listdir(sub_dir_path)
		for item in sub_items:
			if 'OK' in item or 'ok' in item:
				if 'c100' in item or '100' in item:
					NUM_classes = 100
					out_name = item[:10]
				else:
					NUM_classes = 10
					out_name = item[:7]
				print("######processing@@@@@@@", item)
				color_list = copy.deepcopy(color_list_template)
				output_file_name = out_name+"-result("+str(times_for_run)+").gv"
				item_path = os.path.join(sub_dir_path, item)
				with open(os.path.join(item_path, 'nas_config.json')) as f:
					setting = json.load(f)
					spl_times = setting["nas_main"]["spl_network_round"]
					block_num = setting["nas_main"]["block_num"]
					if "repeat_search" in setting["nas_main"].keys():
						repeat_search = setting["nas_main"]["repeat_search"]
					elif "repeat_search" in setting["eva"].keys():
						repeat_search = setting["eva"]["repeat_search"]
					else:
						raise KeyError("can not find the key repeat_search in the nas_config file")
					num_classes = NUM_classes
				input_path = os.path.join(item_path, "nas_log.txt")
				output_path = os.path.join(item_path, output_file_name)
				gen_gv_png(input_path=input_path, output_path=output_path, 
					output_file_name=output_file_name, block_num=block_num, 
					repeat_search=repeat_search, num_classes=num_classes)

