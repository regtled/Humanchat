from openai import OpenAI
import gradio as gr
import os
import re
import json

import asyncio

from collections import deque

import aiohttp
import aiohttp_cors
from aiohttp import web
from aiortc import RTCSessionDescription, RTCPeerConnection
from aiortc.rtcrtpsender import RTCRtpSender

from webrtc import HumanPlayer
import multiprocessing

import logging
logging.basicConfig(level=logging.INFO)

os.environ['HTTP_PROXY'] = f'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = f'http://127.0.0.1:7890'

client = OpenAI()

history = deque(maxlen=6)

digimans = []


def llm_response(message, digiman):
    """
    1.调用OpenAI API返回对话结果并流式返回，对话包含上下文信息\n
    2.对stream结果进行分句，丢给TTS处理\n
    3.防止出现过短的句子，tts处理不自然\n
    4.OpenAI的流式返回为token返回，其他llm模型可能为句子返回，若需要调用其他模型请更改\n
    """
    msgs = list(history)
    message = {"role": "user", "content": message}
    msgs.append(message)
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = msgs,
        temperature = 0,
        top_p = 0,
        stream = True,
        max_tokens=150
    )

    partial_message = ""
    sentence = ""
    for chunk in response:
        print(chunk.choices[0])
        if len(chunk.choices) > 0:
            chunk_message = chunk.choices[0].delta.content
            # print(chunk_message)
            if chunk_message is not None and chunk_message != "":
                clean_message = re.sub(r'[\x00-\x1f\x7f]', '', chunk_message) ## 清理转义字符，防止TTS出现未知bug
                match = re.search(r'[,.?!;:，。？！；：]', clean_message)
                if match:
                    sentence += clean_message[:match.end()] ## OpenAI返回的chunk中可能包含【“，这”】这样的情况，需要分句
                    if len(sentence)>20: ## 防止丢入过短句子
                        # sents.append(sentence)
                        digiman.put_msg(sentence)
                        print("Current sentence: --", sentence)
                        sentence = clean_message[match.end():]
                    else:
                        sentence += clean_message[match.end():]
                else:
                    sentence += clean_message
                partial_message += chunk_message
                # yield partial_message
    if sentence:
        print("Last sentence: --", sentence)
        digiman.put_msg(sentence)
        # sents.append(sentence)
    history.append({"role": "system", "content": partial_message})

pcs = set()

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    sessionid = 0

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
        if pc.connectionState == "closed":
            pcs.discard(pc)

    player = HumanPlayer(digimans[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer)
    capabilities = RTCRtpSender.getCapabilities("video")
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.Response(
        content_type = "application/json",
        text = json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid}
        ),
    )

async def human(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    
    if params.get('interrupt'):
        digimans[sessionid].pause_talk()

    res = await asyncio.get_event_loop().run_in_executor(None, llm_response, params['text'], digimans[sessionid])

    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": "ok"}),
    )

async def humanaudio(request):
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        fileobj = form["file"]
        filebytes = fileobj.file.read()
        digimans[sessionid].put_audio_frame(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "msg": "ok"}),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 1, "msg": "err", "data": ""+e.args[0]+""}),
        )
    
async def set_audiotype(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    digimans[sessionid].set_audio_type(params['type'])
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": "ok"}),
    )

async def is_speaking(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": digimans[sessionid].is_speaking()}),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')

async def run(push_url):
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
        if pc.connectionState == "closed":
            pcs.discard(pc)

    player = HumanPlayer(digimans[0])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    from digiman import MuseDigi
    opt = {"avatar_id": "avator_2", "video_path": "data/digiwoman2.mp4", "bbox_shift": 5, "preparation": False, "batch_size": 16, "sample_rate": 16000, "fps": 50, "l": 10, "r": 10}
    digiman = MuseDigi(opt)
    digimans.append(digiman)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_static("/", "web")

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    for route in list(app.router.routes()):
        cors.add(route)

    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '192.168.1.84', 8010)
        loop.run_until_complete(site.start())
        loop.run_forever()
    run_server(web.AppRunner(app))