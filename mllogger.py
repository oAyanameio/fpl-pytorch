import os
import json
from datetime import datetime


class MLLogger:
    def __init__(self, init=True):
        self.save_dir = None
        self.dir_name = None
        self.log_fn = None

    def initialize(self, root_dir):
        self.save_dir = root_dir
        self.dir_name = root_dir
        self.log_fn = os.path.join(root_dir, "log.txt")
        os.makedirs(self.save_dir, exist_ok=True)

    def info(self, message):
        print(message)

    def get_savedir(self):
        return self.save_dir