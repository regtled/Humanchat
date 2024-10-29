import json
import os
import shutil
import glob
import pickle
import cv2
import torch
from tqdm import tqdm
import sys
import time
import queue
import asyncio
import logging
import copy
import resampy
import numpy as np
import multiprocessing as mp
import soundfile as sf
from io import BytesIO
from tts import OpenAITTS, EdgeTTS
# from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_audio_processor, load_diffusion_model
from av import AudioFrame, VideoFrame
from threading import Thread
from asr import MuseASR
import pdb

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def video2imgs(video_path, output_path, ext = 'png', cut_frame = 10000000):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{output_path}/{count:08d}.{ext}", frame)
            count += 1
        else:
            break

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

def __mirror_index(size, index):
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1

@torch.no_grad()
def inference(render_event, batch_size, latents_out_path, audio_feature_queue, audio_out_queue, res_frame_queue):
    vae, unet, pe = load_diffusion_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)
    pe = pe.half()
    vae.vae = vae.vae.half()
    unet.model = unet.model.half()
    input_latent_list_cycle = torch.load(latents_out_path)
    length = len(input_latent_list_cycle)
    index = 0
    count = 0
    count = 0
    counttime = 0
    logging.info("------Start inference------")
    while True:
        if render_event.is_set():
            # starttime = time.perf_counter()
            try:
                whisper_chunks = audio_feature_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            is_all_silence = True
            audio_frames = []
            for _ in range(batch_size * 2):
                frame, type = audio_out_queue.get()
                audio_frames.append((frame,type))
                if type == 0:
                    is_all_silence = False
            if is_all_silence:
                for i in range(batch_size):
                    res_frame_queue.put((None, __mirror_index(length, index), audio_frames[i*2:i*2+2]))
                    index = index + 1
            else:
                t = time.perf_counter()
                whisper_batch = np.stack(whisper_chunks)
                latent_batch = []
                for i in range(batch_size):
                    idx = __mirror_index(length, index+i)
                    latent = input_latent_list_cycle[idx]
                    latent_batch.append(latent)
                latent_batch = torch.cat(latent_batch, dim=0)
            
                audio_feature_batch = torch.from_numpy(whisper_batch)
                audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype)
                audio_feature_batch = pe(audio_feature_batch)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)

                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample

                recon = vae.decode_latents(pred_latents)

                counttime += (time.perf_counter() - t)
                count += batch_size
                if count >= 100:
                    print(f"--------actual avg infer fps: {count/counttime:.4f}")
                    count = 0
                    counttime = 0
                for i, res_frame in enumerate(recon):
                    res_frame_queue.put((res_frame, __mirror_index(length, index), audio_frames[i*2:i*2+2]))
                    index += 1
        else:
            time.sleep(1)
            
class BaseDigi:
    def __init__(self, opt):
        self.opt = opt

        self.sample_rate = opt["sample_rate"]
        self.fps = opt["fps"]
        self.chunk = self.sample_rate // self.fps

        if opt['tts'] == "openai":
            self.tts = OpenAITTS(opt, self)
        elif opt['tts'] == "edge":
            self.tts = EdgeTTS(opt, self)

        self.curr_state = 0
        self.speaking = False
    
    def put_msg(self, msg):
        self.tts.put_msg(msg)

    def pause_talk(self):
        self.tts.pause_talk()
        self.asr.pause_talk()

    def is_speaking(self)->bool:
        return self.speaking
    
    def put_audio_frame(self, audio_chunk):
        self.asr.put_audio_frame(audio_chunk)

    def put_audio_file(self, filebyte):
        input_stream = BytesIO(filebyte)
        stream = self.process_stream(input_stream)
        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk:
            self.put_audio_frame(stream[idx:idx+self.chunk])
            idx += self.chunk
            streamlen -= self.chunk

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
    
    def set_curr_state(self, audiotype):
        print('Set curr state:', audiotype)
        self.curr_state = audiotype

class MuseDigi(BaseDigi):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt

        self.avatar_id = opt["avatar_id"]
        self.video_path = opt["video_path"]
        self.bbox_shift = opt["bbox_shift"]
        self.avatar_path = f"Avatars/{self.avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": self.avatar_id,
            "video_path": self.video_path,
            "bbox_shift": self.bbox_shift,
        }
        
        self.preparation = opt["preparation"]
        self.batch_size = opt["batch_size"]
        self.res_frame_queue = mp.Queue(self.batch_size*2)
        self.idx = 0
        self.audio_processor = load_audio_processor()
        self.asr = MuseASR(opt, self.audio_processor)
        self.asr.warm_up()
        self.init() ## 加载数字人资源
        self.render_event = mp.Event()
        mp.Process(target=inference, args=(self.render_event, self.batch_size, self.latents_out_path, self.asr.feat_queue, self.asr.output_queue, self.res_frame_queue)).start()

    def init(self):
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)


    # def init(self):
    #     if self.preparation:
    #         if os.path.exists(self.avatar_path):
    #             response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
    #             if response.lower() == "y":
    #                 shutil.rmtree(self.avatar_path)
    #                 print("*********************************")
    #                 print(f"  creating avator: {self.avatar_id}")
    #                 print("*********************************")
    #                 osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_path,self.mask_out_path])
    #                 self.prepare_material()
    #             else:
    #                 self.input_latent_list_cycle = torch.load(self.latents_out_path)
    #                 with open(self.coords_path, 'rb') as f:
    #                     self.coord_list_cycle = pickle.load(f)
    #                 input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    #                 input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    #                 self.frame_list_cycle = read_imgs(input_img_list)
    #                 with open(self.mask_coords_path, 'rb') as f:
    #                     self.mask_coords_list_cycle = pickle.load(f)
    #                 input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
    #                 input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    #                 self.mask_list_cycle = read_imgs(input_mask_list)
    #         else:
    #             print("*********************************")
    #             print(f"  creating avator: {self.avatar_id}")
    #             print("*********************************")
    #             osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_path,self.mask_out_path])
    #             self.prepare_material()
    #     else: 
    #         if not os.path.exists(self.avatar_path):
    #             print(f"{self.avatar_id} does not exist, you should set preparation to True")
    #             sys.exit()

    #         with open(self.avatar_info_path, "r") as f:
    #             avatar_info = json.load(f)
                
    #         if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
    #             response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
    #             if response.lower() == "c":
    #                 shutil.rmtree(self.avatar_path)
    #                 print("*********************************")
    #                 print(f"  creating avator: {self.avatar_id}")
    #                 print("*********************************")
    #                 osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_path,self.mask_out_path])
    #                 self.prepare_material()
    #             else:
    #                 sys.exit()
    #         else:  
    #             self.input_latent_list_cycle = torch.load(self.latents_out_path)
    #             with open(self.coords_path, 'rb') as f:
    #                 self.coord_list_cycle = pickle.load(f)
    #             input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    #             input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    #             self.frame_list_cycle = read_imgs(input_img_list)
    #             with open(self.mask_coords_path, 'rb') as f:
    #                 self.mask_coords_list_cycle = pickle.load(f)
    #             input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
    #             input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    #             self.mask_list_cycle = read_imgs(input_mask_list)
    
    # def prepare_material(self):
    #     print("preparing data materials ... ...")
    #     with open(self.avatar_info_path, "w") as f:
    #         json.dump(self.avatar_info, f)
            
    #     if os.path.isfile(self.video_path):
    #         video2imgs(self.video_path, self.full_imgs_path, ext = 'png')
    #     else:
    #         print(f"copy files in {self.video_path}")
    #         files = os.listdir(self.video_path)
    #         files.sort()
    #         files = [file for file in files if file.split(".")[-1]=="png"]
    #         for filename in files:
    #             shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
    #     input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        
    #     print("extracting landmarks...")
    #     coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
    #     input_latent_list = []
    #     idx = -1
    #     # maker if the bbox is not sufficient 
    #     coord_placeholder = (0.0,0.0,0.0,0.0)
    #     for bbox, frame in zip(coord_list, frame_list):
    #         idx = idx + 1
    #         if bbox == coord_placeholder:
    #             continue
    #         x1, y1, x2, y2 = bbox
    #         crop_frame = frame[y1:y2, x1:x2]
    #         resized_crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    #         from musetalk.models.vae import VAE
    #         vae = VAE(model_path = "./models/sd-vae-ft-mse/")
    #         vae.vae = vae.vae.half()
    #         latents = vae.get_latents_for_unet(resized_crop_frame)
    #         input_latent_list.append(latents)

    #     self.frame_list_cycle = frame_list + frame_list[::-1]
    #     self.coord_list_cycle = coord_list + coord_list[::-1]
    #     self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    #     self.mask_coords_list_cycle = []
    #     self.mask_list_cycle = []

    #     for i,frame in enumerate(tqdm(self.frame_list_cycle)):
    #         cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png",frame)
            
    #         face_box = self.coord_list_cycle[i]
    #         mask,crop_box = get_image_prepare_material(frame,face_box)
    #         cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png",mask)
    #         self.mask_coords_list_cycle += [crop_box]
    #         self.mask_list_cycle.append(mask)
            
    #     with open(self.mask_coords_path, 'wb') as f:
    #         pickle.dump(self.mask_coords_list_cycle, f)

    #     with open(self.coords_path, 'wb') as f:
    #         pickle.dump(self.coord_list_cycle, f)
            
    #     torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))
        
    
    def process_frames(self, quit_event, loop = None, audio_track = None, video_track = None):
        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: ## 两帧音频都是静音
                self.speaking = False
                combine_frame = self.frame_list_cycle[idx]
            else:
                self.speaking = True
                bbox = self.coord_list_cycle[idx]
                ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
                x1, y1, x2, y2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                except:
                    continue
                mask = self.mask_list_cycle[idx]
                mask_crop_box = self.mask_coords_list_cycle[idx]
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            image = combine_frame
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)

            for audio_frame in audio_frames:
                frame, type = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = 16000

                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)
        logging.info('MuseDigi process_frames thread stop')

    def render(self, quit_event, loop = None, audio_track = None, video_track = None):
        self.tts.render(quit_event)
        process_thread = Thread(target=self.process_frames, args=(quit_event, loop, audio_track, video_track))
        process_thread.start()

        self.render_event.set()
        while not quit_event.is_set():
            self.asr.run_step()
            if video_track._queue.qsize()>=1.5*self.opt["batch_size"]:
                print('sleep qsize=', video_track._queue.qsize())
                time.sleep(0.04 * video_track._queue.qsize()*0.8) ##没太看懂
        self.render_event.clear()
        logging.log('MuseDigi thread stop')