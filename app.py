import os
from flask import Flask, request, render_template, flash, redirect, url_for
from Database import *

import Generate_caption
import Generate_caption_1
from Detect_Head_Count import detect
import time
from get_caption import VideoDescriptionRealTime

    
app = Flask(__name__)

global filename
filename = ""

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','avi'])
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/",methods=["POST","GET"])
def index():
   if request.method=="POST":
      f = request.files['userfile']
      path = 'static/{}'.format(f.filename)
      f.save(path)
      caption = Generate_caption.caption_image(path)
      caption1 = Generate_caption_1.caption_image(path)
      total_head = 0
      try:
         total_head = detect(path)[1]['person']
      except Exception as e:
         print(e)
         total_head = 0
      result = {'image':path,'cap':caption,'caption1':caption1,'total_head':total_head}
        
      return render_template('image_index.html',your_caption = result)

   return render_template("index.html", xx= -1)



@app.route("/image",methods=["POST","GET"])
def image():
   if request.method=="POST":
      f = request.files['userfile']
      path = 'static/{}'.format(f.filename)
      f.save(path)
      caption = Generate_caption.caption_image(path)
      caption1 = Generate_caption_1.caption_image(path)
      total_head = 0
      try:
         total_head = detect(path)[1]['person']
      except Exception as e:
         print(e)
         total_head = 0
      result = {'image':path,'cap':caption,'caption1':caption1,'total_head':total_head}
        
      return render_template('image_index.html',your_caption = result)

   return render_template("image_index.html", xx= -1)

 
class config:
    train_path = "data/training_data"
    test_path = "data/testing_data"
    batch_size = 126
    learning_rate = 0.0007
    epochs = 150
    latent_dim = 512
    num_encoder_tokens = 4096
    num_decoder_tokens = 1500
    time_steps_encoder = 80
    max_probability = -1
    save_model_path = 'models'
    validation_split = 0.15
    max_length = 10
    search_type = 'greedy'

 
import os

def remove_files_in_folder(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Iterate over the files and remove each one
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error: {e}")




@app.route("/video",methods=["POST","GET"])
def video():
   if request.method=="POST":
      try:
         remove_files_in_folder("data/testing_data/video")

         f = request.files['userfile']
         path = 'data/testing_data/video/{}'.format(f.filename)
         f.save(path)

         # Call the function to remove files in the specified folder
         video_to_text = VideoDescriptionRealTime(config)
         while True:
            print('.........................\nGenerating Caption:\n')
            start = time.time()
            video_caption, file = video_to_text.test()
            end = time.time()
            sentence = ''
            print(sentence)
            for text in video_caption.split():
                  sentence = sentence + ' ' + text
                  print('\n.........................\n')
                  print(sentence)
            print('\n.........................\n')
            print('It took {:.2f} seconds to generate caption'.format(end-start))
            video_to_text.main(file, sentence)
      except Exception as e:
          print(e)
          import traceback
          traceback.print_exc()
         
   return render_template("video_index.html", xx= -1)


@app.route('/dashboard')
def dashboard():
   return render_template('dashboard.html', xx= -1)

@app.route('/index')
def index1():
   return render_template('index.html')


@app.route('/home')
def home():
   return render_template('index.html')


@app.route('/aboutus')
def aboutus():
   return render_template('aboutus.html')

@app.route('/register',methods = ['POST','GET'])
def registration():
	if request.method=="POST":
		username = request.form["username"]
		email = request.form["email"]
		password = request.form["password"]
		mobile = 0
		InsertData(username,email,password,mobile)
		return render_template('login.html')
		
	return render_template('register.html')


@app.route('/login',methods = ['POST','GET'])
def login():
   if request.method=="POST":
        email = request.form['email']
        passw = request.form['password']
        resp = read_cred(email, passw)
        if resp != None:
            return render_template('image_index.html')
        else:
            message = "Username and/or Password incorrect.\\n        Yo have not registered Yet \\nGo to Register page and do Registration";
            return "<script type='text/javascript'>alert('{}');</script>".format(message)

   return render_template('login.html')

 




if __name__ == "__main__":
    app.run(debug=False)