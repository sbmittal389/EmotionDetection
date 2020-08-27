from flask import Flask, render_template, Response
from camera import VideoCamera
from flask_bootstrap import Bootstrap
import os
app = Flask(__name__)
Bootstrap(app)
#app = Flask(__name__, template_folder='../templates')
@app.route('/home')
@app.route('/')
def index():
    return render_template('home.html')

#The gen() function enters a loop where it continuously returns frames 
# from the camera as response chunks. The function asks the camera to 
# provide a frame by calling the camera.get_frame() method, 
# and then it yields with this frame formatted as a response chunk 
# with a content type of image/jpeg,
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#The generator function used in the /video_feed route is called gen(), 
# and takes as an argument an instance of the Camera class. 
# The mimetype argument is set with the multipart/x-mixed-replace content type(for streaming window button etc.)
# and a boundary set to the string "frame".
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/happy.html')
def hap():
    #songs=os.listdir('static/songs/happy')
    return render_template('happy.html')

@app.route('/sad.html')
def sads():
    #songs=os.listdir('static/songs/happy')
    return render_template('sad.html')

@app.route('/angry.html')
def ang():
    #songs=os.listdir('static/songs/happy')
    return render_template('angry.html')

@app.route('/neutral.html')
def neu():
    #songs=os.listdir('static/songs/happy')
    return render_template('neutral.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)