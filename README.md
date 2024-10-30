# Humanchat
This project is based on Livetalking from https://github.com/lipku/LiveTalking to achieve real-time interactive streaming ditigal man, which realize synchronous audio and video conversations.

## Features
- Simplify the startup method, no need to start the docker service, only use webrtc to push the stream.
- LLM and TTS partially integrate OpenAI services, and you need to add the API key to the environment variables yourself.
- The face-changing part uses musetalk (the visual generation effect and real-time performance are almost currently the best).
- Delay is mainly reflected in calling OpenAI’s TTS service and streaming part.

## Environment Configuration

```
conda create -n humanchat python=3.10
conda activate humanchat
pip install -r requirements.txt
```

## Quick Start
Run the `app.py` script:

```
python app.py
```