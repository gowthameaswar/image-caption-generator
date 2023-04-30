import streamlit as st
import urllib
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from flask import Flask, render_template, request

app = Flask(__name__)

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
max_length = 16
num_beams = 8
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def predict_step(image_path, num_captions):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams, num_return_sequences=num_captions, output_scores=True)
    captions = []
    for output in output_ids:
        preds = tokenizer.decode(output, skip_special_tokens=True)
        preds = preds.strip()
        captions.append(preds)
    return captions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    num_captions = int(request.form['num_captions'])
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        if allowed_file(uploaded_file.filename):
            image = Image.open(uploaded_file)
            image.save(uploaded_file.filename)
            caption = predict_step(uploaded_file.filename, num_captions)
            caption_list = []
            for i in range(num_captions):
                caption_list.append(str(i+1) + '. ' + caption[i])
            return render_template('index.html', caption=caption_list, filename=uploaded_file.filename)
        else:
            return render_template('index.html', message='Invalid file extension. Please upload an image with .jpg, .jpeg, .png, or .jfif extension.')
    else:
        return render_template('index.html', message='No file selected. Please upload an image.')

if __name__ == "__main__":
    app.run(debug=True)
