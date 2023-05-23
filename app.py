from flask import Flask, render_template,request,jsonify,send_file
from flask_socketio import SocketIO, emit
import openai
import json
import os
import subprocess

import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import vits.commons
import vits.utils
from vits.data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence

from scipy.io.wavfile import write


app = Flask(__name__)
socketio = SocketIO(app,cors_allowed_origins='*')

@app.route('/')
def index():
    return render_template('index.html')

openai.api_key = "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"#此处填入API的key

def get_audio(text):
    # 加载模型
    hps = utils.get_hparams_from_file("configs/XXX.json")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint("/path/to/model.pth", net_g, None)

    # 生成音频文件
    wav_file = 'audio.wav'
    stn_tst = get_text(text, hps)
    mel = net_g.generate(torch.from_numpy(stn_tst).cuda())
    utils.save_wav(mel.cpu().numpy(), wav_file, sr=hps.data.sampling_rate)
    return wav_file

@app.route('/chat', methods=['POST'])
def chat():
  # 解析请求数据
  request_data = request.get_json()
  user_message = request_data['messages'][0]['content']
  # 调用 OpenAI API 进行聊天
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": user_message},
    ],
    temperature=0.7,
  )


  # 发送聊天结果和语音给客户端
  socketio.emit('chat_response', response.choices[0].message['content'] )

  # 返回聊天结果
  # return jsonify({'choices': response.choices})

if __name__ == '__main__':
    socketio.run(app, debug=True,port=5001)

