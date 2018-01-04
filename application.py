from flask import Flask, render_template, request, jsonify
import os
import main





application =  app = Flask(__name__)

db_name = 'mydb'
client = None
db = None



# On Bluemix, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run')
def run():
    deviceID = request.args.get('deviceID')
    detect_final_result = main.main_function(deviceID)
    return str(detect_final_result)

if __name__ == '__main__':
    application.run(host='0.0.0.0')
