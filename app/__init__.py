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

# https://media.gettyimages.com/photos/woman-lifts-her-arms-in-victory-mount-everest-national-park-picture-id507910624?s=612x612
# https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/chihuahua-dog-running-across-grass-royalty-free-image-1580743445.jpg

app = Flask(__name__)
app.config['DOWNLOAD_IMAGE'] = "uploaded_download/image/"
app.config['UPLOAD_AUDIO'] = "uploaded_download/caption_audio/"

@app.route('/')
def home():
    useImages = os.listdir(os.path.join(app.static_folder, "gallery"))
    return render_template("index.html", useImages = useImages, error = False)

""" used for uploading image """
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

""" used for gallery image """
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

""" used for URL image """
@app.route('/generate/url', methods=["POST"])
def generate_from_url():
    if request.method == 'POST':
        image_url = request.form['image_url']
        model_to_use = request.form['model_to_use']
        try:
            resp = requests.get(image_url, stream=True)
            if resp.status_code == 200:
                if "jpg" in os.path.basename(image_url) or "png" in os.path.basename(image_url):
                    name_format = datetime.now().strftime("%Y%m%d_%H%M%S%f") + "_" + os.path.basename(image_url)
                else:
                    name_format = datetime.now().strftime("%Y%m%d_%H%M%S%f") + "_" + os.path.basename(image_url) + ".jpg"
                name = os.path.join(app.static_folder, app.config['DOWNLOAD_IMAGE'] + name_format)
                local_file = open(name, 'wb')# Open a local file with wb ( write binary ) permission.
                resp.raw.decode_content = True # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                shutil.copyfileobj(resp.raw, local_file) # Copy the response stream raw data to local image file.
                del resp # Remove the image url response object.
                show_img = app.config['DOWNLOAD_IMAGE'] + name_format
                template_values = generate_caption(name, os.path.splitext(name_format)[0], show_img, model_to_use)
                return render_template("result.html", template_values=template_values)
            else:
                print("URL is not correct")
                useImages = os.listdir(os.path.join(app.static_folder, "gallery"))
                return render_template("index.html", useImages = useImages, error = True)
        except requests.exceptions.RequestException as e:
            print("URL is not correct")
            useImages = os.listdir(os.path.join(app.static_folder, "gallery"))
            return render_template("index.html", useImages = useImages, error = True)


""" generates the caption by calling the pridict function """
def generate_caption(image, audio_filename, show_image_path, model_to_use):
    start_time = time.time()
    name_map = {
        "vgg16": "VGG16",
        "inceptionv3": "InceptionV3",
        "resnet50": "ResNet50",
        "xception": "Xception"
    }
    if model_to_use == "all":
        models = ["xception","vgg16","inceptionv3","resnet50"]
    else:
        models = [model_to_use]
    template_values = {
    "image":show_image_path,
    "itr":{}
    }
    for model in models:
        generateCaption = GenerateCaption(image, model)
        cap_beam, cap_greedy = generateCaption.start()
        "Generating audio for beam search caption"
        audio = gTTS(text=cap_beam, lang='en', slow=False)
        audio_path_beam = os.path.join(app.static_folder, app.config['UPLOAD_AUDIO'] + audio_filename + "_beam_" + model + ".mp3")
        audio.save(audio_path_beam)

        "Generating audio for greedy search caption"
        audio = gTTS(text=cap_greedy, lang='en', slow=False)
        audio_path_greedy = os.path.join(app.static_folder, app.config['UPLOAD_AUDIO'] + audio_filename + "_greedy_" + model + ".mp3")
        audio.save(audio_path_greedy)

        template_values["itr"][model] = {}
        template_values["itr"][model]['name'] = name_map[model]
        template_values["itr"][model]["image"] = show_image_path
        template_values["itr"][model]["cap_beam"] = cap_beam
        template_values["itr"][model]["cap_greedy"] = cap_greedy
        template_values["itr"][model]["audio_path_beam"] = app.config['UPLOAD_AUDIO'] + audio_filename + "_beam_" + model + ".mp3"
        template_values["itr"][model]["audio_path_greedy"] = app.config['UPLOAD_AUDIO'] + audio_filename + "_greedy_" + model + ".mp3"

    print("--- %s seconds ---" % (time.time() - start_time))
    return template_values

@app.route('/models')
def model_comparision():
    return render_template("models.html", template_values=models_summary)
