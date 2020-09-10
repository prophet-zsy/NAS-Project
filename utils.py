import os, sys, queue, time, random, re, json, math, cv2
import numpy as np
import datetime, traceback, pickle
import multiprocessing, copy, signal
from base import Network, NetworkItem, Cell
from info_str import NAS_CONFIG, MF_TEMP

# signal.signal(signal.SIGCHLD,signal.SIG_IGN)


def _dump_stage(stage_info):
    _cur_dir = os.getcwd()
    stage_path = os.path.join(_cur_dir, "memory", "stage_info.pickle")
    with open(stage_path, "w") as f:
        json.dump(stage_info, f, indent=2)


class TimeCnt:
	def __init__(self):
		self.time_stamp = None

	def start(self):
		self.time_stamp = datetime.datetime.now()
		start_time = self.time_stamp.strftime('%d %H:%M:%S')
		return start_time

	def stop(self):
		cost_time = (datetime.datetime.now() - self.time_stamp)
		# format the costtime
		total_seconds = int(cost_time.total_seconds())
		hours = total_seconds//3600
		seconds_inHour = total_seconds%3600
		minutes = seconds_inHour//60
		seconds = seconds_inHour%60
		cost_time = '{}:{}:{}'.format(hours, minutes, seconds)
		return cost_time


class DataSize:
    def __init__(self, eva):
        self.eva = eva
        self.round_count = 0
        self.mode = NAS_CONFIG['nas_main']['add_data_mode']
        #  data size control for game
        self.add_data_per_rd = NAS_CONFIG['nas_main']['add_data_per_round']
        self.init_lr = NAS_CONFIG['nas_main']['init_data_size']
        self.scale = NAS_CONFIG['nas_main']['data_increase_scale']

        self.data_for_confirm_train = NAS_CONFIG['nas_main']['add_data_for_confirm_train']
        self.data_for_retrain = NAS_CONFIG['nas_main']['add_data_for_retrain']

    def _cnt_game_data(self):
        if self.mode == "linear":
            self.round_count += 1
            cur_data_size = self.round_count * self.add_data_per_rd
        elif self.mode == "scale":
            cur_data_size = int(self.init_lr * (self.scale ** self.round_count))
            self.round_count += 1
        else:
            raise ValueError("signal error: mode, it must be one of linear, scale")
        return cur_data_size

    def control(self, stage="game"):
        """Increase the dataset's size in different way

        :param stage: must be one of "game", "confirm"
        :return:
        """
        if stage == "game":
            cur_data_size = self._cnt_game_data()
        elif stage == "confirm":
            cur_data_size = self.data_for_confirm_train
        elif stage == "retrain":
            cur_data_size = self.data_for_retrain
        else:
            raise ValueError("signal error: stage, it must be one of game, confirm")
        if self.eva:
            cur_data_size = self.eva._set_data_size(cur_data_size)
        return cur_data_size


def _epoch_ctrl(eva=None, stage="game"):
    """

    :param eva:
    :param stage: must be one of "game", "confirm", "retrain"
    :return:
    """
    if stage == "game":
        cur_epoch = NAS_CONFIG['eva']['search_epoch']
    elif stage == "confirm":
        cur_epoch = NAS_CONFIG['eva']['confirm_epoch']
    elif stage == "retrain":
        cur_epoch = NAS_CONFIG['eva']['retrain_epoch']
    else:
        raise ValueError("signal error: stage, it must be one of game, confirm, retrain")
    if eva:  # for eva_mask
        eva._set_epoch(cur_epoch)
    return cur_epoch


class EvaScheduleItem:
    def __init__(self, nn_id, alig_id, graph_template, item, pre_blk,\
        ft_sign, bestNN, rd, nn_left, spl_batch_num, epoch, data_size):
        # task content (assign in initial)
        self.nn_id = nn_id
        self.alig_id = alig_id
        self.graph_template = graph_template
        self.network_item = item  # is None when retrain
        self.pre_block = pre_blk
        self.ft_sign = ft_sign
        self.is_bestNN = bestNN
        self.round = rd
        self.nn_left = nn_left
        self.spl_batch_num = spl_batch_num

        self.epoch = epoch
        self.data_size = data_size

        # task info
        self.task_id = -1  # assign in TaskScheduler().exec_task_async
        self.pid = -1  # assgin in task_func
        self.start_time = None # assign in task_func
        self.cost_time = 0 # assign in task_func
        self.gpu_info = -1  # ScheduleItem give gpu to it

        # result
        self.score = 0 # assign in task_func
        self.model_params = 0 # assign in task_func

class PredScheduleItem:
    def __init__(self, net_pool, blk_id):
        self.net_pool = net_pool
        self.blk_id = blk_id
        
        # task info
        self.task_id = -1
        self.pid = -1  # assgin in task_func
        self.start_time = None
        self.cost_time = 0
        self.gpu_info = -1

class TaskScheduler:
    #  Mainly for coordinating GPU resources
    def __init__(self):
        self.task_list = []
        self.result_list = []

        # for multiprocessing communication
        self.result_buffer = multiprocessing.Queue()
        self.signal = multiprocessing.Event()
        
        # store all the running sub process object, and we will use it when make sure the subprocess return
        self.sub_process = {}
        
        # resource
        self.gpu_num = NAS_CONFIG['nas_main']['num_gpu']
        self.gpu_list = queue.Queue()
        for gpu in range(self.gpu_num):
            self.gpu_list.put(gpu)
        # if you put the discrete the id of gpu, please use the following method and comment the two lines above
        # self.gpu_list.put(1)
        
        # for counting task(every task has a unique task_id)
        self.task_id = 0

    def load_tasks(self, tasks):
        self.task_list.extend(tasks)

    def get_task_id(self):
        tmp_id = self.task_id
        self.task_id += 1
        return tmp_id

    def all_alive(self):
        for subpid in self.sub_process.keys():
            if not self.sub_process[subpid].is_alive:
                return False
        return True

    def makesure_sub_return(self, subpid):
        # make sure the sub zombie process vanish
        # pid, exit_code = os.waitpid(subpid, 0)
        # p_wait_vanish = self.sub_process[subpid]
        # p_wait_vanish.join(timeout=10)  # block itself to wait for the subprocess's return
        # if p_wait_vanish.is_alive:
        #     p_wait_vanish.terminate()  # stop it forcely
        #     p_wait_vanish.join()
        del self.sub_process[subpid]

    def exec_task_async(self, task_func, *args, **kwargs):
        """Async: directly return whetherever the tasks is completed
        """
        while self.task_list and not self.gpu_list.empty():
            gpu = self.gpu_list.get()  # get gpu
            task_item = self.task_list.pop(0)  # get task
            # config task
            task_item.gpu_info = gpu
            task_item.task_id = self.get_task_id()
            # exec task
            subp = multiprocessing.Process(target=task_func, args=[task_item, self.result_buffer, self.signal, *args])
            subp.start()
            # sign up the subpid
            self.sub_process[subp.pid] = subp
        self.signal.clear()

    def load_part_result(self):
        """load one or more results if there are tasks completed
        """
        # while self.all_alive():  # when all the subp is alive
        #     self.signal.wait(timeout=100)  # check the subp every 100s
        self.signal.wait()
        # while not self.result_buffer.empty():  # empty() is unreliable!!!
        # while self.result_buffer.qsize() > 0:  # qsize() is unreliable!!!
        while True:
            try:
                task_item = self.result_buffer.get(timeout=2)
            except:
                break
            self.result_list.append(task_item)
            self.gpu_list.put(task_item.gpu_info)  # return gpu
            self.makesure_sub_return(task_item.pid)

    def exec_task(self, task_func, *args, **kwargs):
        """Sync: waiting for all the tasks completed before return
        """
        while self.task_list or self.gpu_list.qsize() < self.gpu_num:
            self.exec_task_async(task_func, *args, **kwargs)
            self.load_part_result()

    def get_result(self):
        result = self.result_list
        self.result_list = []
        return result
#  for test...
def task_fun(task_item, result_buffer, signal, *args, **kwargs):
    import tensorflow as tf
    print("computing gpu {} task {}".format(task_item.gpu_info, task_item.alig_id))
    time.sleep(random.randint(2,20))
    result_buffer.put(task_item)
    signal.set()

class Logger(object):
    def __init__(self):
        _cur_ver_dir = os.getcwd()
        log_dir = os.path.join(_cur_ver_dir, 'memory')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        naslog_path = os.path.join(log_dir, 'nas_log.txt')
        network_info_path = os.path.join(log_dir, 'network_info.txt')
        evalog_path = os.path.join(log_dir, 'evaluator_log.txt')
        errlog_path = os.path.join(log_dir, 'error_log.txt')
        
        self.base_data_dir = os.path.join(log_dir, 'base_data_serialize')

        self._nas_log = open(naslog_path, 'a')
        self._network_log = open(network_info_path, 'a')
        self._eva_log = open(evalog_path, 'a')
        self._error_log = open(errlog_path, 'a')

        self._log_match = {  # match -> log
            'basedata': self.base_data_dir,
            'nas': self._nas_log,
            'net': self._network_log,
            'eva': self._eva_log,
            'err': self._error_log,
            'utils': sys.stdout  # for test
        }

    def __del__(self):
        self._nas_log.close()
        self._network_log.close()
        self._eva_log.close()
        self._error_log.close()

    @staticmethod
    def _get_action(args):
        if isinstance(args, str) and len(args):
            return args, ()
        elif isinstance(args, tuple) and len(args):
            return args[0], args[1:]
        else:
            raise Exception("empty or wrong log args")
        return

    def _log_output(self, match, output, temp, others):
        if not temp:
            assert len(others) == 1, "you must send net to log one by one"
            content = others[0]
            dump_path = os.path.join(self.base_data_dir, "blk_{}_nn_{}.pickle"
                        .format(len(content.pre_block), content.id))
            with open(dump_path, "wb") as f_dump:
                pickle.dump(content, f_dump)
            return
        content = temp.format(others)
        output.write(content)
        output.write('\n')
        if match == "nas":
            print(content)
        if match == "err":
            traceback.print_exc(file=output)
            traceback.print_exc(file=sys.stdout)
        output.flush()
        return

    def __lshift__(self, args):
        """
        Wrtie log or print system information.
        The specified log templeate is defined in info_str.py
        Args:
            args (string or tuple, non-empty)
                When it's tuple, its value is string.
                The first value must be action.
        Return: 
            None
        Example:
            NAS_LOG = Logger() # 'Nas.run' func in nas.py 
            NAS_LOG << 'enuming'
        """
        act, others = Logger._get_action(args)
        match = act.split("_")[0]
        output = self._log_match[match]
        temp = MF_TEMP[act] if match != "basedata" else None
        self._log_output(match, output, temp, others)


NAS_LOG = Logger()


def _check_log():
    _cur_ver_dir = os.getcwd()
    pache_dir = os.path.join(_cur_ver_dir, 'pcache')
    if not os.path.exists(pache_dir):
        os.mkdir(pache_dir)
    log_dir = os.path.join(_cur_ver_dir, 'memory')
    base_data_dir = os.path.join(log_dir, 'base_data_serialize')
    model_dir = os.path.join(_cur_ver_dir, 'model')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        os.mkdir(base_data_dir)
    else:
        if not os.path.exists(base_data_dir):
            os.mkdir(base_data_dir)
        log_dir_sub = os.listdir(log_dir)
        log_files = [os.path.join(log_dir, item) for item in log_dir_sub]
        log_files = [item for item in log_files if os.path.isfile(item)]
        have_content = False
        for file in log_files:
            if os.path.getsize(file) or os.listdir(base_data_dir):
                have_content = True
        if have_content:
            _ask_user(log_files, base_data_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    else:
        _clear_log([], [model_dir])

def _ask_user(log_files, base_data_dir):
    print(MF_TEMP['nas_log_hint'])
    while True:
        # answer = input()
        answer = "y"
        if answer == "n":
            raise Exception(MF_TEMP['nas_existed_log'])
        elif answer == "y":
            log_files = _clear_log(log_files, [base_data_dir])
            break
        else:
            print(MF_TEMP['nas_invalid_str'])

def _clear_log(files, dirs):
    for file in files:
        with open(file, "w") as f:
            f.truncate()
    for dir_ in dirs:
        for item in os.listdir(dir_):
            os.remove(os.path.join(dir_, item))
    return files

# for eva denoise pred huge img, we can cut it into many small pieces, and put them into model in a batch
# one ins for one image
class image_splitor:
    def __init__(self, pieces_len):
        self.pieces_len = pieces_len  # pieces is square
        self.img_h = -1
        self.img_w = -1

    def _split_huge_img(self, data):
        # np data shape is [batch_size, h, w, c], batch_size = 1
        print("spliting the img {}".format(data.shape), flush=True)
        self.img_h = data.shape[1]
        self.img_w = data.shape[2]
        for i in range(0, self.img_h-1, self.pieces_len):
            for j in range(0, self.img_w-1, self.pieces_len):
                end_h = i+self.pieces_len if i+self.pieces_len <= self.img_h else self.img_h
                end_w = j+self.pieces_len if j+self.pieces_len <= self.img_w else self.img_w
                # print(data[:,i:end_h,j:end_w,:].shape)
                data_for_append = np.pad(data[:,i:end_h,j:end_w,:],((0,0),(0,i+100-end_h),(0,j+100-end_w),(0,0)),'constant')
                if i == 0 and j == 0:
                    new_data = data_for_append
                else:
                    new_data = np.concatenate([new_data, data_for_append], axis=0)
        print("into {}".format(new_data.shape), flush=True)
        return new_data

    def _recover_huge_img(self, data):
        # np data shape is [batch_size, h, w, c], batch_size represent many pieces
        print("concating the img {}".format(data.shape), flush=True)
        for i in range(0, self.img_h-1, self.pieces_len):
            for j in range(0, self.img_w-1, self.pieces_len):
                end_h = self.pieces_len if i+self.pieces_len <= self.img_h else self.img_h%self.pieces_len
                end_w = self.pieces_len if j+self.pieces_len <= self.img_w else self.img_w%self.pieces_len
                idx = (i//self.pieces_len)*math.ceil(self.img_w/self.pieces_len)+(j//self.pieces_len)
                # print(data[idx:idx+1,:end_h,:end_w,:].shape)
                if j == 0:
                    new_data_y = data[idx:idx+1,:end_h,:end_w,:]
                else:
                    new_data_y = np.concatenate([new_data_y, data[idx:idx+1,:end_h,:end_w,:]], axis=2)
            if i == 0:
                new_data_x = new_data_y
            else:
                new_data_x = np.concatenate([new_data_x, new_data_y], axis=1)
        print("into {}".format(new_data_x.shape), flush=True)
        return new_data_x


if __name__ == '__main__':
    # NAS_LOG << ('hello', 'I am bread', 'hello world!')
    # NAS_LOG << 'enuming'

    item = NetworkItem(0, [[1,2,3],[4,5,6],[7,8,9]], Cell('conv', 48, 7, 'relu'), [1,0,2,0,1,0])
    tasks = []
    for i in range(15):
        tasks.append(EvaScheduleItem(0, i, [], item, [], False, False, -1, 1, 1))
    TSche = TaskScheduler()
    TSche.load_tasks(tasks)
    TSche.exec_task(task_fun)
    result = TSche.get_result()

    # img  = cv2.imread("./memory/denoise_test/0clear_img.jpg")
    # img = img[np.newaxis,:,:,:]
    # splitor = image_splitor(100)
    # img = splitor._split_huge_img(img)
    # print(img.shape)
    # img = splitor._recover_huge_img(img)
    # print(img.shape)
