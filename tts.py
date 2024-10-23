import queue
from enum import Enum
from threading import Thread
from io import BytesIO
from openai import OpenAI
import soundfile as sf
import numpy as np
import resampy
import asyncio
import logging
logging.basicConfig(level=logging.INFO)

class State(Enum):
    RUNNING=0
    PAUSE=1

class BaseTTS:
    """
    实现一个基础TTS类，提供TTS基础功能，如需加入其他TTS引擎则继承该类
    """
    def __init__(self, opt):
        self.opt = opt

        self.fps = opt["fps"] ## 每秒帧数
        self.sample_rate = opt["sample_rate"] ## 音频采样率
        self.chunk = self.sample_rate // self.fps ## 每帧音频数据长度
        self.input_stream = BytesIO() ## 用于缓存流式返回的音频数据

        self.msgqueue = queue.Queue()
        self.state = State.RUNNING

    def pause_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg(self, msg):
        if len(msg) > 0:
            self.msgqueue.put(msg)
    
    def msg2audio(self, msg):
        pass
    
    def process_msg(self, quit_event):
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            self.msg2audio(msg)
        logging.info("TTS thread exit")

    def render(self, quit_event):
        process_thread = Thread(target=self.process_msg, args=(quit_event,))
        process_thread.start()

class OpenAITTS(BaseTTS):
    def msg2audio(self, msg):
        asyncio.run(self.streamout("tts-1", "nova", msg))
        if self.input_stream.getbuffer().nbytes <= 0:
            logging.warning("OpenAI TTS stream: empty stream, maybe network error")
            return
        self.input_stream.seek(0) ## 重置流指针,指向开头，保证从开头读取音频流式数据
        stream = self.process_stream(self.input_stream) ## 获取当前时刻的音频流快照
        streamlen = stream.shape[0]
        idx = 0

        while streamlen >= self.chunk and self.state == State.RUNNING:
            # self.opt.audio_queue.put(stream[idx:idx+self.chunk])
            idx += self.chunk
            streamlen -= self.chunk
        self.input_stream.seek(0)
        self.input_stream.truncate(0)
        self.input_stream.write(stream[idx:]) ## 有少量数据可能未处理完，为了音频完整性，将剩余数据写入下一次处理，使用一个BytesIO避免内存泄漏的问题

    def process_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream)
        logging.info(f"OpenAI TTS stream: rate {sample_rate}, shape {stream.shape}")
        stream = stream.astype(np.float32)
        if stream.ndim > 1:
            logging.warning(f"OpenAI TTS stream: stream has {stream.ndim} channels, only use the first channel")
            stream = stream[:, 0]
        if sample_rate != self.sample_rate and stream.ndim>0:
            logging.warning(f"OpenAI TTS stream: sample rate {sample_rate} not equal to {self.sample_rate}, resample")
            stream = resampy.resample(stream, sample_rate, self.sample_rate)
        return stream
        
    async def streamout(self, model, voice, text):
        try:
            client = OpenAI()
            with client.audio.speech.with_streaming_response.create(
                model = model,
                voice = voice,
                text = text,
            )as response:
                async for chunk in response.iter_bytes():
                    self.input_stream.write(chunk)
        except Exception as e:
            logging.error(f"Streamout error: {e}")