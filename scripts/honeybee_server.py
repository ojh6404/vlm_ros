#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import numpy as np
import torch
import cv2
from PIL import Image
from flask import Flask, request, jsonify

from pipeline.interface import get_model

# Load trained model
model_path = "checkpoints/7B-C-Abs-M256/last"  # 7B-C-Abs-M144, 7B-D-Abs-M144, 7B-C-Abs-M256
model, tokenizer, processor = get_model(model_path, use_bf16=True, load_in_8bit=True)
model.cuda()
print(f"Model loaded from {model_path}")


def construct_input_prompt(user_prompt):
    # TODO : make it configurable
    SYSTEM_MESSAGE = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    IMAGE_TOKEN = "Human: <image>\n"  # <image> denotes an image placehold.
    USER_PROMPT = f"Human: {user_prompt}\n"
    return SYSTEM_MESSAGE + IMAGE_TOKEN + USER_PROMPT + "AI: "


def infer(query, cvimg, kwargs):
    # images to PIL images
    kwargs["do_sample"] = True if kwargs.get("temperature", 0) > 0 else False
    prompts = [query]
    pilimgs = [Image.fromarray(cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB))]  # TODO : batch
    inputs = processor(text=prompts, images=pilimgs)
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        res = model.generate(**inputs, **kwargs)
    sentences = tokenizer.batch_decode(res, skip_special_tokens=True)
    return sentences


def decode_image(img):
    img = base64.b64decode(img)
    npimg = np.frombuffer(img, dtype=np.uint8)
    cvimg = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return cvimg


# run
if __name__ == "__main__":
    app = Flask(__name__)
    try:

        @app.route("/text_gen", methods=["POST"])
        def text_gen_request():
            data = request.get_json()
            gen_config = data["gen_config"]
            img = data["image"].encode("utf-8")
            cvimg = decode_image(img)
            queries = data["queries"]
            query = construct_input_prompt(queries[0])  # TODO : batch
            sentences = infer(query, cvimg, gen_config)
            response = {"queries": queries, "answeres": sentences}
            return jsonify(response)

    except NameError:
        print("Skipping create text_gen app")

    app.run("0.0.0.0", 8080, threaded=True)
