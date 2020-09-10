from read_net_info import read_network_info, get_y
import os
import math


def chart_gen(net_path, spl_times, blk_num, round_per_blk, num_opt_best, net_num, log_path, 
			  data_scale, data_per_round, init_data_size, data_increase_scale):
	info = read_network_info(net_path)
	y_list, y_list_best, y_sample, y_sample_best, y_list_average, tmp_str = get_y(info, spl_times)

	round_best_socre = []
	for blk in y_list_best:
		for rd in range(round_per_blk):
			best_score = 0
			for net in blk:
				if net[0] > best_score:
					best_score = net[0]
				if rd == round_per_blk - 1:
					if net[1] > best_score:
						best_score = net[1]
				net.pop(0)
			while [] in blk:
				blk.remove([])
			round_best_socre.append(best_score)

	round_best_socre = [eval('{:.4f}'.format(i)) for i in round_best_socre]
	print(round_best_socre)

	with open(log_path) as f:
		lines = f.readlines()
		round_time = []
		time_per_net = []
		for line in lines:
			if "NAS: The round is over, cost time:" in line or "NAS: Train winner finished and cost time:" in line:
				round_time.append(float(line.split(" ")[-1]))
			if "arrange_result:" in line:
				try:
					time_per_net.append(float(line.split(":")[-1]))
				except:  # the eva process of the network suffer some problem
					print("there is some error in one of eva process")
					time_per_net.append(time_per_net[-1])

		chart = []
		index = 0  # to compute averge time for every net
		chart_str = ""
		for i in range(len(round_best_socre)):
			one_line = []
			if (i+1) % round_per_blk == 1:
				rd_net_num = net_num
			if (i+1) % round_per_blk == 0:
				spl_ts = num_opt_best
			else:
				spl_ts = spl_times
			rd_num = i%round_per_blk+1
			one_line.append('rd'+str(rd_num))
			if data_scale == "linear":
				data_added = data_per_round*(i+1)
			elif data_scale == "linear_block":  # if add data in block mode
				data_added = data_per_round*(i//round_per_blk+1)
			elif data_scale == "scale":
				data_added = int(init_data_size * (data_increase_scale**i))
			else:
				raise ValueError("data_scale must be one of linear and scale")
			if data_added > 40000:
				data_added = 40000
			one_line.append(str(data_added))
			one_line.append(str(round_best_socre[i]))
			one_line.append(str(rd_net_num))
			one_line.append(str(int(round_time[i])))
			item_num = rd_net_num * spl_ts
			# print(index, item_num)
			# print(rd_num, time_per_net[index: index + item_num])
			ave_net_time = sum(time_per_net[index: index + item_num])/item_num
			# if (i+1) % round_per_blk == 0:
			# 	ave_net_time = 0
			one_line.append(str(int(ave_net_time)))
			one_line.append(str(spl_ts))
			acclerate_ratio = ave_net_time * item_num / int(round_time[i])
			one_line.append('{:.2f}'.format(acclerate_ratio))

			rd_net_num = math.ceil(rd_net_num/2)

			index += item_num

			one_line_str = "|"
			for item in one_line:
				one_line_str += item
				one_line_str += "|"
			one_line_str += "\n"
			# print(one_line_str)
			chart_str += one_line_str
			chart.append(one_line_str)
		return chart_str

if __name__ == '__main__':
	path = "D:/personal_zhangshuyu/my_document/论文/NAS/CompetitionNAS/Experiment/NAS_0111/NAS_zsy/NAS-c100-5_OK_0.693"

	data_scale = "linear"
	data_per_round = 1600
	init_data_size = 1600
	data_increase_scale = 1.2
	log_path = os.path.join(path, "nas_log.txt")
	blk_num = 4
	round_per_blk = 10
	net_num = 286
	num_opt_best = 12

	net_path = os.path.join(path, "network_info.txt")
	spl_times = 3

	chart_gen(net_path, spl_times, blk_num, round_per_blk, num_opt_best, net_num, log_path, 
			  data_scale, data_per_round, init_data_size, data_increase_scale)