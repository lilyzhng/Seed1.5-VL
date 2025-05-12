# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import os
import gradio as gr
from infer import SeedVLInfer, ConversationModeI18N, ConversationModeCN

infer = SeedVLInfer(api_key=os.environ.get('API_KEY'))

label_translations = {
    "gr_chatinterface_ofl": {
        "English": "Chatbot",
        "中文": "对话界面"
    },
    "gr_chatinterface_ol": {
        "English": "Chatbot",
        "中文": "对话界面"
    },
    "gr_tab_ol": {
        "English": "Online",
        "中文": "在线模式"
    },
    "gr_tab_ofl": {
        "English": "Offline",
        "中文": "离线模式"
    },
    "gr_thinking": {
        "English": ConversationModeI18N.D,
        "中文": ConversationModeCN.D,
    },
    "gr_temperature": {
        "English": "Temperature",
        "中文": "温度系数"
    },
    "gr_webcam_image": {
        "English": "🤳 Open Webcam",
        "中文": "🤳 打开摄像头"
    },
    "gr_webcam_images": {
        "English": "📹 Recorded Frames",
        "中文": "📹 录制的视频帧"
    },
    "gr_chatinterface_ofl.textbox.placeholder": {
        "English":
        "Ask me anything. You can also drop in images and .mp4 videos.",
        "中文": "有什么想问的？支持上传图片和.mp4视频。"
    },
    "gr_chatinterface_ol.textbox.placeholder": {
        "English": "Ask me anything...",
        "中文": "有什么想问的？"
    }
}


def offline_chat(gr_inputs: dict, gr_history: list, infer_history: list,
                 if_thinking: bool, temperature: float):
    mode = ConversationModeI18N.D if if_thinking else ConversationModeI18N.G
    for response_text, infer_history in infer(inputs=gr_inputs,
                                              history=infer_history,
                                              mode=mode,
                                              temperature=temperature):
        if if_thinking:
            reasoning_text, response_text = response_text.split('</think>')
            reasoning_text = reasoning_text.lstrip('<think>')
            response_message = [{
                "role": "assistant",
                "content": reasoning_text,
                'metadata': {
                    'title': '🤔 Thinking'
                }
            }, {
                "role": "assistant",
                "content": response_text
            }]
            yield response_message, infer_history
        else:
            response_message = [{
                "role": "assistant",
                "content": response_text
            }]
            yield response_text, infer_history


def online_record_chat(text: str, gr_history: list, gr_webcam_images: list,
                       gr_counter: int, infer_history: list, if_thinking: bool,
                       temperature: float):
    if not gr_webcam_images:
        gr_webcam_images = []
    gr_webcam_images = gr_webcam_images[gr_counter:]
    inputs = {'text': text, 'files': [webp for webp, _ in gr_webcam_images]}
    yield f'received {len(gr_webcam_images)} new frames, processing...', gr_counter + len(
        gr_webcam_images), infer_history
    for response_message, infer_history in offline_chat(
            inputs, gr_history, infer_history, if_thinking, temperature):
        yield response_message, gr.skip(), infer_history


with gr.Blocks() as demo:
    with gr.Column():
        gr_title = gr.Markdown('# Seed1.5-VL')
        with gr.Row():
            gr_lang_selector = gr.Dropdown(choices=["English", "中文"],
                                           value="English",
                                           label="🌐 English Interface/中文界面",
                                           interactive=True,
                                           min_width=250,
                                           scale=0)
    with gr.Tabs():
        with gr.Tab("Offline") as gr_tab_ofl:
            gr_infer_history = gr.State([])
            gr_thinking_hidden = gr.Checkbox(value=True, visible=False)
            gr_temperature_hidden = gr.Slider(minimum=0.0,
                                              maximum=2.0,
                                              step=0.1,
                                              value=1.0,
                                              interactive=True,
                                              visible=False)
            gr_chatinterface_ofl = gr.ChatInterface(
                fn=offline_chat,
                type="messages",
                multimodal=True,
                textbox=gr.MultimodalTextbox(
                    file_count="multiple",
                    file_types=["image", ".mp4"],
                    sources=["upload"],
                    stop_btn=True,
                    placeholder=label_translations[
                        'gr_chatinterface_ofl.textbox.placeholder']['English'],
                ),
                additional_inputs=[
                    gr_infer_history, gr_thinking_hidden, gr_temperature_hidden
                ],
                additional_outputs=[gr_infer_history],
            )
            gr.on(triggers=[gr_chatinterface_ofl.chatbot.clear],
                  fn=lambda: [],
                  outputs=[gr_infer_history])
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    gr_thinking_ofl = gr.Checkbox(
                        value=True,
                        label=label_translations['gr_thinking']['English'],
                    )
                    gr_thinking_ofl.change(lambda x: x,
                                           inputs=gr_thinking_ofl,
                                           outputs=gr_thinking_hidden)
                    gr_temperature_ofl = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        label=label_translations['gr_temperature']['English'],
                        interactive=True)
                    gr_temperature_ofl.change(lambda x: x,
                                              inputs=gr_temperature_ofl,
                                              outputs=gr_temperature_hidden)
                with gr.Column(scale=8):
                    with gr.Column(visible=True) as gr_examples_en:
                        gr.Examples(
                            examples=[
                                {
                                    "text": "Who are you?",
                                    "files": []
                                },
                                {
                                    "text": "Introduce this.",
                                    "files": ["examples/bancopy.jpg"]
                                },
                                {
                                    "text":
                                    """Find Curry's "Good Night" celebration time.""",
                                    "files":
                                    ["examples/I7pTpMjqNRM_1080p_small.mp4"]
                                },
                                {
                                    "text":
                                    "Share your feelings.",
                                    "files": [
                                        "examples/newyork.jpg",
                                        "examples/beijing.jpg"
                                    ]
                                },
                            ],
                            inputs=[gr_chatinterface_ofl.textbox],
                        )
                    with gr.Column(visible=False) as gr_examples_cn:
                        gr.Examples(
                            label='示例',
                            examples=[
                                {
                                    "text": "你是谁？",
                                    "files": []
                                },
                                {
                                    "text": "介绍一下。",
                                    "files": ["examples/bancopy.jpg"]
                                },
                                {
                                    "text":
                                    "找到库里的“晚安”庆祝时间段。",
                                    "files":
                                    ["examples/I7pTpMjqNRM_1080p_small.mp4"]
                                },
                                {
                                    "text":
                                    "你有什么感想？",
                                    "files": [
                                        "examples/newyork.jpg",
                                        "examples/beijing.jpg"
                                    ]
                                },
                            ],
                            inputs=[gr_chatinterface_ofl.textbox],
                        )
        with gr.Tab("Online") as gr_tab_ol:
            with gr.Row():
                with gr.Column(scale=1):
                    gr_infer_history = gr.State([])
                    gr_thinking_hidden = gr.Checkbox(value=True, visible=False)
                    gr_temperature_hidden = gr.Slider(minimum=0.0,
                                                      maximum=2.0,
                                                      step=0.1,
                                                      value=1.0,
                                                      interactive=True,
                                                      visible=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr_webcam_image = gr.Image(
                                label=label_translations['gr_webcam_image']
                                ['English'],
                                sources="webcam",
                                height=250,
                                type='filepath')
                            gr_webcam_images = gr.Gallery(
                                label=label_translations['gr_webcam_images']
                                ['English'],
                                show_label=True,
                                format='webp',
                                columns=1,
                                height=250,
                                preview=True,
                                interactive=False)
                            gr_counter = gr.Number(value=0, visible=False)
                        with gr.Column(scale=3):
                            gr_chatinterface_ol = gr.ChatInterface(
                                fn=online_record_chat,
                                type="messages",
                                multimodal=False,
                                textbox=gr.
                                Textbox(placeholder=label_translations[
                                    'gr_chatinterface_ol.textbox.placeholder']
                                        ['English'],
                                        submit_btn=True,
                                        stop_btn=True),
                                additional_inputs=[
                                    gr_webcam_images, gr_counter,
                                    gr_infer_history, gr_thinking_hidden,
                                    gr_temperature_hidden
                                ],
                                additional_outputs=[
                                    gr_counter, gr_infer_history
                                ],
                            )

                            def cache_webcam(recorded_image: str,
                                             recorded_images: list):
                                if not recorded_images:
                                    recorded_images = []
                                return recorded_images + [recorded_image]

                            gr_webcam_image.stream(
                                fn=cache_webcam,
                                inputs=[gr_webcam_image, gr_webcam_images],
                                outputs=[gr_webcam_images],
                                stream_every=1,
                                concurrency_limit=30,
                            )
                            with gr.Row():
                                gr_thinking_ol = gr.Checkbox(
                                    value=True,
                                    label=label_translations['gr_thinking']
                                    ['English'],
                                )
                                gr_thinking_ol.change(
                                    lambda x: x,
                                    inputs=gr_thinking_ol,
                                    outputs=gr_thinking_hidden)
                                gr_temperature_ol = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    step=0.1,
                                    value=1.0,
                                    label=label_translations['gr_temperature']
                                    ['English'],
                                    interactive=True)
                                gr_temperature_ol.change(
                                    lambda x: x,
                                    inputs=gr_temperature_ol,
                                    outputs=gr_temperature_hidden)

    def update_lang(lang: str):
        return (
            gr.update(label=label_translations['gr_chatinterface_ofl'][lang]),
            gr.update(label=label_translations['gr_chatinterface_ol'][lang]),
            gr.update(placeholder=label_translations[
                'gr_chatinterface_ofl.textbox.placeholder'][lang]),
            gr.update(placeholder=label_translations[
                'gr_chatinterface_ol.textbox.placeholder'][lang]),
            gr.update(label=label_translations['gr_tab_ofl'][lang]),
            gr.update(label=label_translations['gr_tab_ol'][lang]),
            gr.update(label=label_translations['gr_thinking'][lang]),
            gr.update(label=label_translations['gr_thinking'][lang]),
            gr.update(label=label_translations['gr_temperature'][lang]),
            gr.update(label=label_translations['gr_temperature'][lang]),
            gr.update(visible=lang == 'English'),
            gr.update(visible=lang != 'English'),
            gr.update(label=label_translations['gr_webcam_image'][lang]),
            gr.update(label=label_translations['gr_webcam_images'][lang]),
        )

    gr_lang_selector.change(fn=update_lang,
                            inputs=[gr_lang_selector],
                            outputs=[
                                gr_chatinterface_ofl.chatbot,
                                gr_chatinterface_ol.chatbot,
                                gr_chatinterface_ofl.textbox,
                                gr_chatinterface_ol.textbox,
                                gr_tab_ofl,
                                gr_tab_ol,
                                gr_thinking_ofl,
                                gr_thinking_ol,
                                gr_temperature_ofl,
                                gr_temperature_ol,
                                gr_examples_en,
                                gr_examples_cn,
                                gr_webcam_image,
                                gr_webcam_images,
                            ])
demo.queue(default_concurrency_limit=100, max_size=100).launch(share=True,
                                                               max_threads=100,
                                                               ssr_mode=False)
