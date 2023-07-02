from flask import Flask, render_template, request
# from keras.models import load_model
# from keras.preprocessing import image
from sequential_process_v2 import sequential_process
import webbrowser

app = Flask(__name__)


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p, a = sequential_process(img_path, 'output_')
		pred = f"Prediction: {p['confidence']}% {p['label']}"
		sev_score = f"Infected Area: {p['severity']} %"
		ses_class = f"SES Class: {p['SES class']}"

	return render_template(
		"index.html", 
		prediction = pred, 
		sev = sev_score, 
		ses_class = ses_class, 
		img_path = a)


if __name__ =='__main__':
    # Open the browser with the app's URL
    webbrowser.open('http://192.168.0.105:5000/')
    # Start the Flask app
    app.run(host='192.168.0.105', debug = True)