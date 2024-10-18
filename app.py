from openai import OpenAI
import gradio as gr
import os
import re

os.environ['HTTP_PROXY'] = f'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = f'http://127.0.0.1:7890'

client = OpenAI()
sents =[]

def llm_response(message, history):
    """
    1.调用OpenAI API返回对话结果并流式返回，对话包含上下文信息
    2.对stream结果进行分句，丢给TTS处理
    3.防止出现过短的句子，tts处理不自然
    4.OpenAI的流式返回为token返回，其他llm模型可能为句子返回，若需要调用其他模型请更改
    """
    history_openai_format = []
    for msg in history:
        msg = {"role": msg["role"], "content": msg["content"]}
        history_openai_format.append(msg)
    message = {"role": "user", "content": message}
    history_openai_format.append(message)
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = history_openai_format,
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
                    if len(sentence)>10: ## 防止丢入过短句子
                        sents.append(sentence)
                        sentence = clean_message[match.end():]
                    else:
                        sentence += clean_message[match.end():]
                else:
                    sentence += clean_message
                partial_message += chunk_message
                yield partial_message
    if sentence:
        sents.append(sentence)
    print(sents)
                
gr.ChatInterface(llm_response, type="messages").launch(server_name="192.168.1.84", server_port=8010)