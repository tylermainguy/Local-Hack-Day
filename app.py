import os
from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image

UPLOAD_FOLDER = 'C:/Users/bunny/Desktop/qhacks/Local-Hack-Day/user images'
ALLOWED_EXTENSIONS = set([ 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# dimensions of our images
img_width, img_height = 224, 224
model = load_model('lhd.h5')
global graph
graph = tf.get_default_graph() 

#model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

img = image.load_img('static/img/orange_14.jpg', target_size=(img_width, img_height))
img_tensor = image.img_to_array(img)
#add a dimension 
img_tensor = np.expand_dims(img_tensor, axis=0)


@app.route("/")
def home():
    return render_template('index.html')

@app.route('/indexImages', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('indexImages.html')
        elif (allowed_file(file.filename)) == False:
            return redirect('indexInvalidImages')

@app.route('/indexInvalidImages')
def invalid_image():
    return render_template('indexInvalidImages.html')

@app.route("/classify", methods=['POST'])	
def classify():
	#if request.method == 'POST':
		#f = request.files['the_file']
		with graph.as_default():
			data = {"success": False}

			#result = model.predict(img_tensor)

			y_prob = model.predict(img_tensor) 
			y_class = y_prob.argmax(axis=-1)

			#return the classification label
			if (y_class == np.array([0])):
				lbl = "Apple " + str(y_prob[0][0])
			elif(y_class == np.array([1])):
				lbl = "Banana " + str(y_prob[0][1])
			elif(y_class == np.array([2])):
				lbl = "Orange " + str(y_prob[0][2])

			#takes the folders names in alphabetical order to assign labels to indices. 
			#Apple maps to 0, Banana maps to 1, Orange maps to 2


			#data["prediction"] = str(model.predict(img_tensor))
			data["prediction"] = lbl
			data["success"] = True
            #resultprob = model.predict_proba(img_tensor)
			

		return jsonify(data)

if __name__ == "__main__":
    app.run()
