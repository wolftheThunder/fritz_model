from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
import random
import argparse
import threading
import copy
import torch.multiprocessing as mp
from aiohttp import web
import aiohttp_cors

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender
import openai

# Ensure TensorRT is imported correctly
try:
    import tensorrt as trt
except ImportError:
    print("TensorRT is not installed. Please install it if you need to use TensorRT models.")

nerfreals = {}
pcs = set()

latest_llm_reply = {}

def llm_response(message, nerfreal):
    """
    Calls the OpenAI ChatCompletion API and sends the reply to the container.
    The reply is stored in the global latest_llm_reply dictionary using the session ID as key.
    """
    global latest_llm_reply
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        reply = response.choices[0].message["content"].strip()
        session_id = str(nerfreal.opt.sessionid)
        latest_llm_reply[session_id] = reply
        print(f"LLM reply for session {session_id}: {reply}")
        if hasattr(nerfreal, "put_msg_txt"):
            nerfreal.put_msg_txt(reply)
    except Exception as e:
        print("LLM error:", str(e))

def randN(N):
    """Generate a random integer with N digits."""
    return random.randint(10**(N-1), 10**N - 1)

def build_nerfreal(sessionid):
    """
    Build and return a container instance based on the specified model.
    Instead of modifying the global opt, we create a deep copy of it for this session.
    """
    session_opt = copy.deepcopy(opt)
    session_opt.sessionid = sessionid

    if session_opt.model.lower() == 'wav2lip':
        from lipreal import LipReal, load_model, load_avatar, warm_up
        model_instance = load_model(session_opt.model_path)
        avatar_instance = load_avatar("wap2lip384avatar1")
        warm_up(session_opt.max_session, model_instance, 384)
        nerfreal = LipReal(session_opt, model_instance, avatar_instance)
    elif session_opt.model.lower() == 'musetalk':
        from musereal import MuseReal, load_model, load_avatar, warm_up
        model_instance = load_model()
        avatar_instance = load_avatar("avator_1")
        warm_up(session_opt.max_session, model_instance)
        nerfreal = MuseReal(session_opt, model_instance, avatar_instance)
    elif session_opt.model.lower() == 'ernerf':
        from nerfreal import NeRFReal, load_model, load_avatar
        model_instance = load_model(session_opt)
        avatar_instance = load_avatar(session_opt)
        nerfreal = NeRFReal(session_opt, model_instance, avatar_instance)
    elif session_opt.model.lower() == 'ultralight':
        from lightreal import LightReal, load_model, load_avatar, warm_up
        model_instance = load_model(session_opt)
        avatar_instance = load_avatar("avator_1")
        warm_up(session_opt.max_session, avatar_instance, 160)
        nerfreal = LightReal(session_opt, model_instance, avatar_instance)
    elif session_opt.model.lower() == 'tensorrt':
        from tensor_rt import TensorRTModel  # Ensure TensorRT is imported
        model_instance = TensorRTModel(session_opt.model_path)
        nerfreal = TensorRTModel(session_opt, model_instance)
    else:
        raise ValueError("Unknown model: " + session_opt.model)
    return nerfreal

async def offer(request):
    """
    Handles the /offer endpoint: receives an SDP offer, creates a container,
    and returns an SDP answer along with a session ID.
    """
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    if len(nerfreals) >= opt.max_session:
        return web.Response(text="-1")
    sessionid = randN(6)
    nerfreals[sessionid] = None
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            nerfreals.pop(sessionid, None)

    if opt.transport.lower() == "mediasoup":
        player = MediasoupHumanPlayer(nerfreals[sessionid], opt.mediasoup_url)
    else:
        player = HumanPlayer(nerfreals[sessionid])

    pc.addTrack(player.audio)
    pc.addTrack(player.video)
    capabilities = RTCRtpSender.getCapabilities("video")
    prefs = [c for c in capabilities.codecs if c.name in ["H264", "VP8", "rtx"]]
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(prefs)
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "sessionid": sessionid
    })

async def human(request):
    """
    Handles text messages via the /human endpoint.
    Expects a JSON payload with keys: sessionid, type, text, and optionally interrupt.
    """
    params = await request.json()
    sessionid = params.get("sessionid")
    if sessionid is None or sessionid not in nerfreals or nerfreals[sessionid] is None:
        return web.json_response({"code": -1, "data": "Invalid session id"}, status=400)
    if params.get("interrupt") and hasattr(nerfreals[sessionid], "flush_talk"):
        nerfreals[sessionid].flush_talk()
    if params["type"] == "echo" and hasattr(nerfreals[sessionid], "put_msg_txt"):
        nerfreals[sessionid].put_msg_txt(params["text"])
    elif params["type"] == "chat":
        await asyncio.get_event_loop().run_in_executor(
            None, llm_response, params["text"], nerfreals[sessionid]
        )
    return web.json_response({"code": 0, "data": "ok"})

async def humanaudio(request):
    """Handles audio file uploads via /humanaudio."""
    try:
        form = await request.post()
        sessionid = int(form.get("sessionid", -1))
        if sessionid not in nerfreals or nerfreals[sessionid] is None:
            return web.json_response({"code": -1, "msg": "Invalid session id"}, status=400)
        filebytes = form["file"].file.read()
        if hasattr(nerfreals[sessionid], "put_audio_file"):
            nerfreals[sessionid].put_audio_file(filebytes)
        return web.json_response({"code": 0, "msg": "ok"})
    except Exception as e:
        return web.json_response({"code": -1, "msg": "err", "data": str(e)})

async def set_audiotype(request):
    """Handles /set_audiotype to update container state."""
    params = await request.json()
    sessionid = params.get("sessionid")
    if sessionid is None or sessionid not in nerfreals or nerfreals[sessionid] is None:
        return web.json_response({"code": -1, "data": "Invalid session id"}, status=400)
    if hasattr(nerfreals[sessionid], "set_curr_state"):
        nerfreals[sessionid].set_curr_state(params["audiotype"], params["reinit"])
    return web.json_response({"code": 0, "data": "ok"})

async def record(request):
    """Handles start/stop recording commands via /record."""
    params = await request.json()
    sessionid = params.get("sessionid")
    if sessionid is None or sessionid not in nerfreals or nerfreals[sessionid] is None:
        return web.json_response({"code": -1, "data": "Invalid session id"}, status=400)
    if params["type"] == "start_record" and hasattr(nerfreals[sessionid], "start_recording"):
        nerfreals[sessionid].start_recording()
    elif params["type"] == "end_record" and hasattr(nerfreals[sessionid], "stop_recording"):
        nerfreals[sessionid].stop_recording()
    return web.json_response({"code": 0, "data": "ok"})

async def is_speaking(request):
    """Returns whether the container is currently speaking."""
    params = await request.json()
    sessionid = params.get("sessionid")
    if sessionid is None or sessionid not in nerfreals or nerfreals[sessionid] is None:
        return web.json_response({"code": -1, "data": "Invalid session id"}, status=400)
    speaking = (nerfreals[sessionid].is_speaking() if hasattr(nerfreals[sessionid], "is_speaking") else False)
    return web.json_response({"code": 0, "data": speaking})

async def llm_reply_handler(request):
    """
    GET endpoint to return the latest LLM reply for the specified session.
    The session ID is expected as a query parameter.
    """
    sessionid = request.query.get("sessionid", None)
    if sessionid is None:
        return web.json_response({"reply": ""})
    reply = latest_llm_reply.get(sessionid, "")
    return web.json_response({"reply": reply})

async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="wav2lip",
                        help="Model type: ernerf, musetalk, wav2lip, ultralight, tensorrt")
    parser.add_argument("--transport", type=str, default="mediasoup",
                        help="Transport: rtmp, rtcpush, mediasoup, aiortc")
    parser.add_argument("--mediasoup_url", type=str, default="ws://localhost:3000",
                        help="Mediasoup signaling URL")
    parser.add_argument("--max_session", type=int, default=3,
                        help="Max concurrent sessions")
    parser.add_argument("--listenport", type=int, default=8010,
                        help="HTTP server port")
    parser.add_argument("--push_url", type=str,
                        default="http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream",
                        help="RTMP/RTCPush URL")
    parser.add_argument("--fps", type=int, default=50,
                        help="Frames per second for audio chunk calculation")
    parser.add_argument("--model_path", type=str, default="./models/wav2lip.pth",
                        help="Path to the model checkpoint file")
    parser.add_argument("--tts", type=str, default="edgetts",
                        help="TTS engine to use: edgetts, gpt-sovits, xtts, cosyvoice, fishtts")
    parser.add_argument("--l", type=int, default=160,
                        help="Stride left size for ASR module")
    parser.add_argument("--r", type=int, default=160,
                        help="Stride right size for ASR module")
    opt = parser.parse_args()

    # Pre-load model for non-WebRTC transports if needed.
    if opt.model.lower() == "wav2lip":
        from lipreal import LipReal, load_model, load_avatar, warm_up
        model = load_model(opt.model_path)
        avatar = load_avatar("wap2lip384avatar1")
        warm_up(opt.max_session, model, 384)
    elif opt.model.lower() == "musetalk":
        from musereal import MuseReal, load_model, load_avatar, warm_up
        model = load_model()
        avatar = load_avatar("avator_1")
        warm_up(opt.max_session, model)
    elif opt.model.lower() == "ernerf":
        from nerfreal import NeRFReal, load_model, load_avatar
        model = load_model(opt)
        avatar = load_avatar(opt)
    elif opt.model.lower() == "ultralight":
        from lightreal import LightReal, load_model, load_avatar, warm_up
        model = load_model(opt)
        avatar = load_avatar("avator_1")
        warm_up(opt.max_session, avatar, 160)
    elif opt.model.lower() == "tensorrt":
        from tensor_rt import TensorRTModel  # Ensure TensorRT is imported
        model = TensorRTModel(opt.model_path)  # Initialize TensorRT model
    else:
        raise ValueError("Unknown model: " + opt.model)

    # For RTMP transport (not used for mediasoup)
    if opt.transport.lower() == "rtmp":
        thread_quit = threading.Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = threading.Thread(target=lambda: nerfreals[0].render(thread_quit))
        rendthrd.start()

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_get("/llm_reply", llm_reply_handler)
    app.router.add_static("/", path="web")

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True, expose_headers="*", allow_headers="*"
        )
    })
    for route in list(app.router.routes()):
        cors.add(route)

    pagename = "webrtcapi.html"
    if opt.transport.lower() == "rtmp":
        pagename = "echoapi.html"
    elif opt.transport.lower() == "rtcpush":
        pagename = "rtcpushapi.html"
    print(f"Server starting at http://<server_ip>:{opt.listenport}/{pagename}")
    runner = web.AppRunner(app)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "0.0.0.0", opt.listenport)
    loop.run_until_complete(site.start())
    loop.run_forever()