"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import streamlit as st
import subprocess

# Install required libraries
libs = ['opencv-python-headless', 'torch', 'torchvision']
for lib in libs:
    try:
        subprocess.call(['pip', 'install', lib])
    except:
        st.write(f'Could not install {lib}')


import argparse
import io
import os
from PIL import Image
import datetime

import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 16

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"


@app.route("/", methods=["GET", "POST"])
def predict():
    
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()  # updates results.imgs with boxes and labels
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/{now_time}.png"
        Image.fromarray(results.ims[0]).save(img_savename)
        return redirect(img_savename)

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    #model = torch.hub.load('ultralytics/yolov5', 'custom', r'C:\Users\fa19-bcs-040.cuiwah\Desktop\yolo checking\yolov5\runs\train\yolov5s_results\weights/best.pt')

    import torch
    import urllib.request

    url = "https://raw.githubusercontent.com/ZeeshanKhalid2k01/weed_web_app/main/best.pt"
    urllib.request.urlretrieve(url, "best.pt")
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')

    
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
