from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from kmeanApi import  *
import os
# initialize flask application
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))
cors = CORS(app)
@app.route('/api/train', methods=['GET'])
@cross_origin()
def train():
    # get parameters from request
    objmsapi = msApi ();
    accuracy = objmsapi.trainModel (request.args.get ('split_Size'))
    print (accuracy)
    return jsonify ({'accuracy': 'created'})

@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():
    # get iris object from request
    objmsapi = msApi ()
    criteria = request.json
    cluster = objmsapi.predict (criteria)
    b=cluster.tolist()
    print(b[0])
    print( type(b[0]))
    return jsonify ({'cluster': b[0] } )

@app.route("/", methods=["GET"])
@cross_origin()
def all():
    return 'python Server Working'
if __name__ == '__main__':
    # run web server
    app.run (debug=True , host='0.0.0.0' , port=port)
