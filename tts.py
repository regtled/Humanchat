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
import edge_tts


class State(Enum):
    RUNNING=0
    PAUSE=1

class BaseTTS:
    """
    实现一个基础TTS类，提供TTS基础功能，如需加入其他TTS引擎则继承该类
    """
    def __init__(self, opt, parent):
        self.opt = opt
        self.parent = parent ## parent参数代表这个TTS绑定的是哪个数字人实例

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
                self.state = State.RUNNING
            except queue.Empty:
                continue
            self.msg2audio(msg)
        logging.info("TTS thread exit")

    def render(self, quit_event):
        process_thread = Thread(target=self.process_msg, args=(quit_event,))
        process_thread.start()

class OpenAITTS(BaseTTS):
    '''
    1. 部署OpenAI TTS的流式输出，在使用时请阅读最新文档或查看issue，保证函数可用(with_streaming_response())\n
    2. 由于返回的格式受限，Opus、AAC、FLAC、WAV、PCM、MP3，含有该格式的头部信息，所以末尾可能会出现少量不足一帧的音频会被丢弃的现象。影响不大\n
    '''
    def msg2audio(self, msg):
        asyncio.new_event_loop().run_until_complete(self.streamout("tts-1", "nova", msg))
        if self.input_stream.getbuffer().nbytes <= 0:
            logging.warning("OpenAI TTS stream: empty stream, maybe network error or something else!")
            return
        self.input_stream.seek(0) ## 重置流指针,指向开头，保证从开头读取音频流式数据（包含wav头部信息）
        stream = self.process_stream(self.input_stream) ## 获取当前时刻的音频流快照
        streamlen = stream.shape[0]
        idx = 0

        while streamlen >= self.chunk and self.state == State.RUNNING:
            # self.opt.audio_queue.put(stream[idx:idx+self.chunk])
            self.parent.put_audio_frame(stream[idx:idx+self.chunk])
            idx += self.chunk
            streamlen -= self.chunk
        self.input_stream.seek(0)
        self.input_stream.truncate()
        # self.input_stream.write(stream[idx:]) ## 有少量数据可能未处理完，为了音频完整性，将剩余数据写入下一次处理，使用一个BytesIO避免内存泄漏的问题
        ## OpenAI TTS的返回结果为封装好的音频格式，比如wav，有头部信息，只写入末尾数据会有问题！！wav头部信息占44字节，期望让这44字节一直留在input_stream里

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
        
    async def streamout(self, model, voice, input):
        try:
            client = OpenAI()
            with client.audio.speech.with_streaming_response.create(
                model = model,
                voice = voice,
                input = input,
                response_format = "wav",
                speed = 1,
            )as response:
                first_chunk = True
                for chunk in response.iter_bytes():
                    if first_chunk:
                        first_chunk = False
                    self.input_stream.write(chunk)
        except Exception as e:
            logging.error(f"Streamout error: {e}")

class EdgeTTS(BaseTTS):
    '''
    1. 部署了微软Edge TTS的流式输出，按字节流的形式输出，再用soundfile库读取字节流，分chunk传给parent数字人实例\n
    2. 使用Edge TTS是联网服务，需要保证网络通畅；且确保edge_tts库是最新版本的\n
    addition: edge_tts在中国刚刚被墙了（2024.10.30），需要挂代理\n
    '''
    def msg2audio(self, msg):
        asyncio.new_event_loop().run_until_complete(self.streamout("zh-CN-YunxiaNeural", msg))
        if self.input_stream.getbuffer().nbytes <= 0:
            logging.warning("Edge TTS stream: empty stream, maybe network error or something else!")
            return
        self.input_stream.seek(0) ## 重置流指针,指向开头，保证从开头读取音频流式数据（包含wav头部信息）
        stream = self.process_stream(self.input_stream) ## 获取当前时刻的音频流快照
        streamlen = stream.shape[0]
        idx = 0

        while streamlen >= self.chunk and self.state == State.RUNNING:
            # self.opt.audio_queue.put(stream[idx:idx+self.chunk])
            self.parent.put_audio_frame(stream[idx:idx+self.chunk])
            idx += self.chunk
            streamlen -= self.chunk
        self.input_stream.seek(0)
        self.input_stream.truncate()

    def process_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream)
        logging.info(f"Edge TTS stream: rate {sample_rate}, shape {stream.shape}")
        stream = stream.astype(np.float32)
        if stream.ndim > 1:
            logging.warning(f"Edge TTS stream: stream has {stream.ndim} channels, only use the first channel")
            stream = stream[:, 0]
        if sample_rate != self.sample_rate and stream.ndim>0:
            logging.warning(f"Edge TTS stream: sample rate {sample_rate} not equal to {self.sample_rate}, resample")
            stream = resampy.resample(stream, sample_rate, self.sample_rate)
        return stream
        
    async def streamout(self, voice, input):
        try:
            communicate = edge_tts.Communicate(input, voice, proxy="http://127.0.0.1:7890")
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    self.input_stream.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logging.error(f"Streamout error: {e}")