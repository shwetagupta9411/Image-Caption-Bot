from flask import Flask, render_template, request, url_for, redirect
from generate_caption import GenerateCaption
from werkzeug.utils import secure_filename
from flask import send_from_directory
from config import models_summary
from datetime import datetime
from gtts import gTTS
import requests
import shutil
import time
import os


# https://www.dev2qa.com/demo/images/green_button.jpg
# http://www.pngmart.com/files/7/Red-Smoke-Transparent-Images-PNG.png
# https://media.gettyimages.com/photos/woman-lifts-her-arms-in-victory-mount-everest-national-park-picture-id507910624?s=612x612

app = Flask(__name__)
app.config['DOWNLOAD_IMAGE'] = "uploaded_download/image/"
app.config['UPLOAD_AUDIO'] = "uploaded_download/caption_audio/"

@app.route('/')
def home():
    useImages = os.listdir(os.path.join(app.static_folder, "galary"))
    return render_template("index.html", useImages = useImages)

@app.route('/generate', methods=["POST"])
def generate():
    if request.method == 'POST':
        image = request.files['image']
        model_to_use = request.form['model_to_use']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        filename = secure_filename(timestamp + "_" + image.filename)
        image_name = os.path.join(app.static_folder, app.config['DOWNLOAD_IMAGE'] + filename)
        image.save(image_name)
        filename_audio = timestamp + "_" + os.path.splitext(image.filename)[0]
        show_image = app.config['DOWNLOAD_IMAGE'] + filename
        template_values = generate_caption(image_name, filename_audio, show_image, model_to_use)
        return render_template("result.html", template_values=template_values)

@app.route('/generate/gallery', methods=["POST"])
def generate_from_gallery():
    if request.method == 'POST':
        image = request.form['image']
        model_to_use = request.form['model_to_use']
        image_file = os.path.join(app.static_folder, image)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        filename_audio = timestamp + "_" + os.path.splitext(image.split("/")[1])[0]
        template_values = generate_caption(image_file, filename_audio, image, model_to_use)
        return render_template("result.html", template_values=template_values)

@app.route('/generate/url', methods=["POST"])
def generate_from_url():
    if request.method == 'POST':
        image_url = request.form['image_url']
        model_to_use = request.form['model_to_use']
        resp = requests.get(image_url, stream=True)
        if resp.status_code == 200:
            name_format = datetime.now().strftime("%Y%m%d_%H%M%S%f") + "_" + os.path.basename(image_url)
            name = os.path.join(app.static_folder, app.config['DOWNLOAD_IMAGE'] + name_format)
            local_file = open(name, 'wb')# Open a local file with wb ( write binary ) permission.
            resp.raw.decode_content = True # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            shutil.copyfileobj(resp.raw, local_file) # Copy the response stream raw data to local image file.
            del resp # Remove the image url response object.
            show_img = app.config['DOWNLOAD_IMAGE'] + name_format
            template_values = generate_caption(name, os.path.splitext(name_format)[0], show_img, model_to_use)
            return render_template("result.html", template_values=template_values)
        else:
            print("file not found")


def generate_caption(image, audio_filename, show_image_path, model_to_use):
    start_time = time.time()
    name_map = {
        "vgg16": "VGG16",
        "inceptionv3": "InceptionV3",
        "rasnet50": "RasNet50",
        "xception": "Xception"
    }
    if model_to_use == "all":
        models = ["vgg16","inceptionv3","rasnet50","xception"]
    else:
        models = [model_to_use]
    template_values = {}
    # cap = "hello shweta"
    print(models)
    for model in models:
        generateCaption = GenerateCaption(image, model)
        cap = generateCaption.start()
        audio = gTTS(text=cap, lang='en', slow=False)
        audio_path = os.path.join(app.static_folder, app.config['UPLOAD_AUDIO'] + audio_filename + "_" + model + ".mp3")
        audio.save(audio_path)

        template_values[model] = {}
        template_values[model]['name'] = name_map[model]
        template_values[model]["image"] = show_image_path
        template_values[model]["cap"] = cap
        template_values[model]["audio_path"] = app.config['UPLOAD_AUDIO'] + audio_filename + "_" + model + ".mp3"

    print("--- %s seconds ---" % (time.time() - start_time))
    return template_values

@app.route('/models')
def model_comparision():
    return render_template("models.html", template_values=models_summary)
