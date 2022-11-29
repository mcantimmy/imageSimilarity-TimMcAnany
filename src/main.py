import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from numpy import asarray
import bz2
import pickle

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

#allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#form to upload image
@app.route('/')
def upload_form():
    return render_template('upload.html')

#upload image and find similar images from model
@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected')
		return redirect(request.url)
	if file.filename not in os.listdir('static/uploads'):
		flash('Image not from dataset')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		img = Image.open(file)
		img = img.resize((100,100), Image.BILINEAR)
		img = asarray(img)
		feature = img / 255
		feature = feature.flatten()
		feature = feature.reshape(1, -1)
		with bz2.BZ2File('isknn.pkl', 'rb') as f:
			img_model = pickle.load(f)

		distances, nbors = img_model.kneighbors(feature)
		distances, nbors = distances[0], nbors[0]
		imgs = os.listdir('static/uploads')
		imgs.sort()
		filenames = [{} for _ in range(len(nbors))]
		for s,i in enumerate(nbors):
			filenames[s]['filename'] = imgs[i]
			# we get euclidean distance, to convert to similarity we do 1 - l1 norm of the distance
			if s == 0:
				filenames[s]['similarity'] = "Input"
			else:
				filenames[s]['similarity'] = f"Similarity: {100 * (1 - (distances[s] / sum(distances))):.0f}%"
			print(filenames)
		flash('Product has been evaluated and similar products are displayed below')
		return render_template('upload.html', filenames=filenames)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

#displays images from dataset
@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
	app.run()