import numpy as np
import config


class LoD:
    def __init__(self, max_resolution, decimals=5):
        self.value = config.initial_lod
        self.max_value = int(np.log2(max_resolution)) - 2
        self.decimals = decimals
        self.max_resolution = max_resolution
        self.resolution = config.initial_res

    def get_value(self):
        return np.round(np.float32(self.value), decimals=self.decimals)

    def reached_max(self):
        return self.value >= self.max_value

    def increase_value(self, steps):
        increment = 1 / steps
        increment = increment * (10 ** self.decimals)
        increment = np.floor(increment)
        increment = increment / (10 ** self.decimals)
        if self.value < self.max_value:
            self.value = self.value + increment

    def get_resolution(self):
        return self.resolution

    def increase_resolution(self):
        self.resolution = min(self.resolution * 2, self.max_resolution)

    def round(self):
        self.value = np.round(self.value, decimals=0)
