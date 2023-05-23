# pyopenjtalk 的安装
# https://www.bilibili.com/video/BV13t4y1V7DV/?vd_source=f17bac2fc1c6cdda2557b1601f2c6413


from flask import Flask, render_template,request,jsonify,send_file
from flask_socketio import SocketIO, emit
import openai
import json
import os
import subprocess
from scipy.io import wavfile
# import math
import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# 帮助 Python 找到 vits 目录下的文件
import sys
sys.path.append(os.path.dirname(__file__) + '\\vits')
print(sys.path)
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

openai.api_key = ""#填入openai的key
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')


def get_text(text, hps):
    """
    将文本规范化
    """
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

@app.route('/')
def index():
    return render_template('index.html')

def get_audio(text):
    # 加载模型
    hps = utils.get_hparams_from_file(r'./vits/dlmodel/config.json')
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    utils.load_checkpoint(r"./vits/dlmodel/keqing.pth", net_g, None)
    _ = net_g.eval().to(torch.device('cpu'))

    # 生成音频文件，并将音频保存在 static 目录下
    wav_file = './static/audio.wav'
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.to(torch.device('cpu')).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(torch.device('cpu'))
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    # ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))    # 在notebook上面进行显示
    wavfile.write(wav_file, rate=hps.data.sampling_rate, data=audio)
    return audio    #audio 是一个 numpy 数组

@app.route('/chat/', methods=['POST'])
def chat():
    # 解析请求数据
    request_data = request.get_json()
    user_message = request_data['messages'][0]['content']
    
    # # 调用 OpenAI API 进行聊天
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": user_message},
    #     ],
    #     temperature=0.7,
    # )

    # # 生成语音
    # audio_file = get_audio('[ZH]' + response.choices[0].text + '[ZH]')
    audio_file = get_audio(user_message)

    # # 发送聊天结果和语音给客户端
    # socketio.emit('chat_response', {'text': response.choices[0].text, 'audio_file': audio_file})
    # socketio.emit('chat_response', response.choices[0].message['content'] )
    socketio.emit('chat_response', user_message)

    # 返回聊天结果
    # return jsonify({'choices': response.choices})
    return None

if __name__ == '__main__':
    socketio.run(app, debug=True,port=5001)
