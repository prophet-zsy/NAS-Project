import time
import random
from multiprocessing import Process,Event,Queue

def gpu_compute(e, item, gpu, result):
	with open("temp.txt", "a", encoding="utf-8") as f:
		f.write("computing gpu {} task {}\n".format(gpu, item))
		print("computing gpu {} task {}".format(gpu, item))
	time.sleep(random.randint(2,20))
	result.put((item, gpu))
	e.set()

if __name__ == '__main__':
	e = Event()
	task = [i for i in range(15)]
	result = Queue()
	gpu_q = [i for i in range(4)]
	p_list = []
	while task or len(gpu_q) is not 4:
		while task and gpu_q:
			gpu = gpu_q.pop(0)
			item = task.pop(0)
			p = Process(target=gpu_compute,args=[e, item, gpu, result])
			p.start()
			p_list.append(p)
		e.clear()
		# while True:
		# 	for p in p_list:
		# 		if not p.is_alive:
		# 			break
		e.wait(timeout=20)
		#  这里应该加一个p.join()或者os.waitpid()来回收执行完毕后进入僵尸进程状态的子进程
		#  <僵尸状态的子进程不一定会被自动回收，据查资料再次调用p.start()会回收，但这样最后一个调用的子进程便不能被回收>
		#  <这里gpu_compute函数过于简单，所以测试问题，但是复杂场景下，僵尸进程便会跳出来叨扰你>
		with open("temp.txt", "a", encoding="utf-8") as f:
			f.write("host is waked up\n")
			print("host is waked up")
		while not result.empty():
			score, gpu = result.get()
			with open("temp.txt", "a", encoding="utf-8") as f:
				f.write("item {}, gpu{} finished\n".format(score, gpu))
				print("item {}, gpu{} finished".format(score, gpu))
			gpu_q.append(gpu)
