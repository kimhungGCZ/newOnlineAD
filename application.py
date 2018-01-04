# from cloudant import Cloudant
# from flask import Flask, render_template, request, jsonify
# import atexit
# import cf_deployment_tracker
# import os
# import json
# import main
#
#
# # Emit Bluemix deployment event
# cf_deployment_tracker.track()
#
# application =  app = Flask(__name__)
#
# db_name = 'mydb'
# client = None
# db = None
#
# if 'VCAP_SERVICES' in os.environ:
#     vcap = json.loads(os.getenv('VCAP_SERVICES'))
#     print('Found VCAP_SERVICES')
#     if 'cloudantNoSQLDB' in vcap:
#         creds = vcap['cloudantNoSQLDB'][0]['credentials']
#         user = creds['username']
#         password = creds['password']
#         url = 'https://' + creds['host']
#         client = Cloudant(user, password, url=url, connect=True)
#         db = client.create_database(db_name, throw_on_exists=False)
# elif os.path.isfile('vcap-local.json'):
#     with open('vcap-local.json') as f:
#         vcap = json.load(f)
#         print('Found local VCAP_SERVICES')
#         creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
#         user = creds['username']
#         password = creds['password']
#         url = 'https://' + creds['host']
#         client = Cloudant(user, password, url=url, connect=True)
#         db = client.create_database(db_name, throw_on_exists=False)
#
# # On Bluemix, get the port number from the environment variable PORT
# # When running this app on the local machine, default the port to 8000
# port = int(os.getenv('PORT', 8000))
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
# @app.route('/run')
# def run():
#     deviceID = request.args.get('deviceID')
#     detect_final_result = main.main_function(deviceID)
#     return str(detect_final_result)
#
# # /* Endpoint to greet and add a new visitor to database.
# # * Send a POST request to localhost:8000/api/visitors with body
# # * {
# # *     "name": "Bob"
# # * }
# # */
# @app.route('/api/visitors', methods=['GET'])
# def get_visitor():
#     if client:
#         return jsonify(list(map(lambda doc: doc['name'], db)))
#     else:
#         print('No database')
#         return jsonify([])
#
# # /**
# #  * Endpoint to get a JSON array of all the visitors in the database
# #  * REST API example:
# #  * <code>
# #  * GET http://localhost:8000/api/visitors
# #  * </code>
# #  *
# #  * Response:
# #  * [ "Bob", "Jane" ]
# #  * @return An array of all the visitor names
# #  */
# @app.route('/api/visitors', methods=['POST'])
# def put_visitor():
#     user = request.json['name']
#     if client:
#         data = {'name':user}
#         db.create_document(data)
#         return 'Hello %s! I added you to the database.' % user
#     else:
#         print('No database')
#         return 'Hello %s!' % user
#
# @atexit.register
# def shutdown():
#     if client:
#         client.disconnect()
#
# if __name__ == '__main__':
#     application.run(host='0.0.0.0', port=port, debug=True)
import flask
import os

application = flask.Flask(__name__)

# Only enable Flask debugging if an env var is set to true
application.debug = os.environ.get('FLASK_DEBUG') in ['true', 'True']

# Get application version from env
app_version = os.environ.get('APP_VERSION')

# Get cool new feature flag from env
enable_cool_new_feature = os.environ.get('ENABLE_COOL_NEW_FEATURE') in ['true', 'True']


@application.route('/')
def hello_world():
    message = "Hello, world!"
    return flask.render_template('index.html',
                                 title=message,
                                 flask_debug=application.debug,
                                 app_version=app_version,
                                 enable_cool_new_feature=enable_cool_new_feature)


if __name__ == '__main__':
    application.run(host='0.0.0.0')