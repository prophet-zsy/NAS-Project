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
	while task or len(gpu_q) is not 4:
		while task and gpu_q:
			gpu = gpu_q.pop(0)
			item = task.pop(0)
			Process(target=gpu_compute,args=[e, item, gpu, result]).start()
		e.clear()
		e.wait()
		with open("temp.txt", "a", encoding="utf-8") as f:
			f.write("host is waked up\n")
			print("host is waked up")
		while not result.empty():
			score, gpu = result.get()
			with open("temp.txt", "a", encoding="utf-8") as f:
				f.write("item {}, gpu{} finished\n".format(score, gpu))
				print("item {}, gpu{} finished".format(score, gpu))
			gpu_q.append(gpu)
