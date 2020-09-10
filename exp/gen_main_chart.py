import os


def gen_chart(report_path):
	chart_str = "**概览**  \n\n"
	chart_str += "| exp | blk_num | repeat | graph | skipping(dist/num) | num_opt_best | spl_net_rd | result |\n"
	chart_str += "|----| -------- | -----  | ----- |  ----              |     -----  |  ----------- | ------ |\n"
	with open(report_path, "r", encoding='utf-8') as f:
		contents = f.readlines()
	
	items = []
	for line in contents:
		if line.startswith("### "):
			exp_id = line.replace("### ", "").replace("\n", "")
			items.append("[{}](#{})".format(exp_id, exp_id.lower()))
		if line.startswith("        \"block_num\":"):
			blk_num = line.split(" ")[-1].replace(",", "").replace("\n", "")
			items.append(blk_num)
		if line.startswith("            \"depth\":"):
			depth = line.split(" ")[-1].replace(",", "").replace("\n", "")
			items.append(depth)
		if line.startswith("            \"width\":"):
			width = line.split(" ")[-1].replace(",", "").replace("\n", "")
			items.append(width)
		if line.startswith("            \"max_depth\":"):
			max_depth = line.split(" ")[-1].replace(",", "").replace("\n", "")
			items.append(max_depth)
		if line.startswith("        \"skip_max_dist\":"):
			skip_max_dist = line.split(" ")[-1].replace(",", "").replace("\n", "")
			items.append(skip_max_dist)
		if line.startswith("        \"skip_max_num\":"):
			skip_max_num = line.split(" ")[-1].replace(",", "").replace("\n", "")
			items.append(skip_max_num)
		if line.startswith("        \"num_opt_best\":"):
			num_opt_best = line.split(" ")[-1].replace(",", "").replace("\n", "")
			items.append(num_opt_best)
		if line.startswith("        \"spl_network_round\":"):
			spl_net_rd = line.split(" ")[-1].replace(",", "").replace("\n", "")
			items.append(spl_net_rd)
		if line.startswith("        \"repeat_search\":"):
			repeat = line.split(" ")[-1].replace(",", "").replace("\n", "")
			items.append(repeat)
		if line.startswith("0."):
			result = line.split(" ")[0].replace(",", "").replace("\n", "")
			items.append(result)
	
	assert len(items) % 11 == 0, "there must be something wrong when reading the report, please check it !!!"

	new_items = []
	for i in range(int(len(items)/11)):
		base_id = i*11
		exp = []
		exp.append(items[base_id+0])
		exp.append(items[base_id+1])
		exp.append(items[base_id+9])
		graph = items[base_id+2]+items[base_id+3]+items[base_id+4]
		exp.append(graph)
		skip = items[base_id+5]+"/"+items[base_id+6]
		exp.append(skip)
		exp.append(items[base_id+7])
		exp.append(items[base_id+8])
		exp.append(items[base_id+10])
		new_items.append(exp)

	for exp in new_items:
		chart_line = "|"
		for item in exp:
			chart_line += item
			chart_line += "|"
		chart_line += "\n"
		chart_str += chart_line
	chart_str += "\n"
	print(chart_str)
	return chart_str


if __name__ == '__main__':
	report_dir = os.path.join(os.getcwd(), "exp_report", "exp_repo_restruct")
	report_name = "report01_13.md"
	report_path = os.path.join(report_dir, report_name)
	main_chart = gen_chart(report_path)

	main_chart_lines = main_chart.split("\n")
	for i in range(len(main_chart_lines)):
		main_chart_lines[i] += "\n"
	main_chart_lines[-1].replace("\n", "")

	with open(report_path, "r", encoding='utf-8') as f:
		contents = f.readlines()
	main_chart_lines.insert(0, contents[0])
	main_chart_lines.extend(contents[1:])
	contents = main_chart_lines
	with open(report_path, "w", encoding='utf-8') as f:
		f.writelines(contents)
