import numpy as np

import queue
from queue import Queue
import multiprocessing as mp
from musetalk.whisper.audio2feature import Audio2Feature

class BaseASR:
    def __init__(self, opt):
        self.opt = opt

        self.fps = opt.fps
        self.sample_rate = opt.sample_rate
        self.chunk = self.sample_rate // self.fps
        self.queue = Queue() ## 缓存的音频帧
        self.output_queue = mp.Queue()

        self.batch_size = opt.batch_size

        self.frames = []
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r

        self.feat_queue = mp.Queue(2)

    def pause_talk(self):
        self.queue.queue.clear()

    def put_audio_frame(self, audio_chunk):
        self.queue.put(audio_chunk)

    def get_audio_frame(self):
        """
        type 0: 有音频帧\n
        type 1: 无音频帧, 静音\n
        addition: 若嘴角还是有抖动异常，可加小于一定值手动设置为闭嘴\n
        """
        try:
            frame = self.queue.get(block=True, timeout=0.01)
            type = 0
        except queue.Empty:
            frame = np.zeros(self.chunk, dtype=np.float32)
            type = 1
        return frame, type
    
    def is_audio_frame_empty(self):
        return self.queue.empty()
    
    def get_audio_out(self):
        """
        把原始音频帧传递给数字人
        """
        return self.output_queue.get()
    
    def run_step(self):
        pass

    def get_next_feat(self, block, timeout):
        return self.feat_queue.get(block=block, timeout=timeout)
    
    def warm_up(self):
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame, type = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame, type))
        for _ in range(self.stride_left_size):
            self.output_queue.get()
    
class MuseASR(BaseASR):
    def __init__(self, opt, audio_processor:Audio2Feature):
        super().__init__(opt)
        self.audio_processor = audio_processor

    def run_step(self):
        for _ in range(self.batch_size*2): ## 音频帧率为视频的两倍
            audio_frame, type = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame, type))

        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames)
        whisper_feature = self.audio_processor.audio2feat(inputs)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array = whisper_feature, fps = self.fps/2, batch_size = self.batch_size, start = self.stride_left_size/2)
        self.feat_queue.put(whisper_chunks)
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):] ## 保留一定数量的音频帧作为历史消息（没看懂，但能用）