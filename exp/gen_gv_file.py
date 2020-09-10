import os, copy, json
from graph_subtraction import subtraction
color_list_template = ['yellow', 'green', 'blue', 'orange', 'red', 'cyan', 'pink', 'gold', 'purple', 'firebrick', 
				'olive', 'tomato', 'slategray', 'teal', 'black', 'sienna', 'silver', 'skyblue']


def read_graph(input_path, block_num):
	with open(input_path, "r") as f:
		content = f.readlines()
		graph = content[-1-block_num:-1]
		graph = "".join(graph)
		graph = "[" + graph
		graph = graph[:-1]
		graph += "]"
		graph = eval(graph)
	return graph

def repeat_block(graph, repeat_search):
	new_graph = []
	for blk in graph:
		for _ in range(repeat_search):
			blk_cp = copy.deepcopy(blk)
			blk_cp[0].append([])  # add tail node
			blk_cp[1].append(('identity'))
			new_graph.append(blk_cp)
		new_graph[-1][1][-1] = ('pooling', 'max', 2)
	return new_graph

def gen_gv_code(graph, num_classes=None, view_mode="whole_map", mark_edge=None):
	"""
	args: view_mode (whole_map or block)
	     in whole_map, num_classes is used, in block, num_classes is not used
	"""
	color_list = copy.deepcopy(color_list_template)
	code_for_write = "digraph G {\n"
	# input
	code_for_write += "    0[style=solid,color={},shape=box,label=\"input\"];\n\n".format(color_list.pop(0))
	# blocks
	for blk_id in range(len(graph)):
		code_for_write += "    subgraph cluster_{} ".format(blk_id)
		code_for_write += "{\n"
		code_for_write += "    color=gray;\n"
		code_for_write += "    node [style=solid,color={},shape=box];\n".format(color_list.pop(0))
		for cell_id in range(len(graph[blk_id][1])):
			code_for_write += "    {}{}[label=\"{}\"];\n".format(blk_id+1, cell_id, graph[blk_id][1][cell_id])
		code_for_write += "    label = \"Block{}\";\n".format(blk_id+1)
		code_for_write += "    }\n\n"
	# output
	last_color = color_list.pop(0)
	if view_mode == "whole_map":
		code_for_write += "    1[style=solid,color={},shape=box,label=\"('fc', 256, 'relu')\"];\n".format(last_color)
		code_for_write += "    2[style=solid,color={},shape=box,label=\"('fc', {}, None)\"];\n\n".format(last_color, num_classes)
	elif view_mode == "block":
		code_for_write += "    1[style=solid,color={},shape=box,label=\"output\"];\n".format(last_color)
	# connect
	code_for_write += "    0 -> 10\n\n"
	for blk_id in range(len(graph)):
		for node_id in range(len(graph[blk_id][0])):
			code_for_write += "    {}{} -> ".format(blk_id+1, node_id)
			if graph[blk_id][0][node_id]:  # if there is any node in the list
				for out_node in graph[blk_id][0][node_id]:
					if not mark_edge or \
						mark_edge and out_node not in mark_edge[blk_id][0][node_id]:  # mask the marked edge
						code_for_write += "{}{},".format(blk_id+1, out_node)
				code_for_write = code_for_write[:-1]
				# mark the spcified edge
				if mark_edge and mark_edge[blk_id][0][node_id]:
					code_for_write += "    {}{} -> ".format(blk_id+1, node_id)
					for out_node in mark_edge[blk_id][0][node_id]:
						code_for_write += "{}{},".format(blk_id+1, out_node)
					code_for_write = code_for_write[:-1]  # remove the ","
					code_for_write += " [color=red]"
			else:  # if it is the last node in the block
				if blk_id == len(graph)-1:  # if the block is the last block
					code_for_write += "1"
				else:
					code_for_write += "{}{}".format(blk_id+2, 0)
			code_for_write += "\n"
		code_for_write += "\n"
	if view_mode == "whole_map":
		code_for_write += "    1 -> 2\n"
	code_for_write += "}\n"
	return code_for_write

def gv_code_to_png(gv_code, output_path):
	gv_file = output_path + ".gv"
	with open(gv_file, "w") as f:
		f.write(gv_code)
	output_file_png = output_path + ".png"
	os.system("dot -Tpng -o "+output_file_png+" "+str(gv_file))

def gen_gv_png(input_path, output_path, block_num, repeat_search, num_classes):
	graph = read_graph(input_path, block_num)
	graph = repeat_block(graph, repeat_search)
	# print(graph)
	gv_code = gen_gv_code(graph, num_classes, view_mode="whole_map")
	# print(gv_code)
	gv_code_to_png(gv_code, output_path)

def blk_to_png(graph, output_path, mark_skip_edge=[]):
	gv_code = gen_gv_code(graph, view_mode="block", mark_edge=mark_skip_edge)
	gv_code_to_png(gv_code, output_path)

if __name__ == '__main__':
	times_for_run = 7

	dir_path = os.path.join(os.getcwd(), 'NAS_0113')
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
				output_file_name = out_name+"-result("+str(times_for_run)+")"
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
				gen_gv_png(input_path=input_path, output_path=output_path, block_num=block_num, 
							repeat_search=repeat_search, num_classes=num_classes)

