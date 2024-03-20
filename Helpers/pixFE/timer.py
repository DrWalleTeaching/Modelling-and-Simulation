
import time
class Timer:   

    def __init__(self):
        self.start_times={}
        self.stop_times={}
        self.elapsed_times={}

    def start(self,name):
        self.start_times[name]=time.time()
        self.elapsed_times[name]=0
    def stop(self,name):
        self.stop_times[name]=time.time()
        self.elapsed_times[name]+=-self.start_times[name]+self.stop_times[name]
    def elapsed(self,name):
        return self.elapsed_times[name]