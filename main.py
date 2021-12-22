from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from db import FaissDatabase
from models.swin_transformer import extract_feature
import os

UPLOAD_DIR = "data/tmp"
IMAGE_DB_DIR = "data/image_db"

app = Flask(__name__)
CORS(app, resources=r'/*')
db = FaissDatabase(index_path="checkpoints/db_index.idx",
                   meta_path="checkpoints/db_meta.pkl")


@app.route("/")
def index():
    return jsonify(msg="A simple search engine base on Swin-Transformer and Faiss")


@app.route("/add", methods=['GET', 'POST'])
def add():
    pass


@app.route("/img/<filename>", methods=['GET'])
def get_image(filename):
    with open(os.path.join(IMAGE_DB_DIR, filename), 'rb') as f:
        return Response(f.read(), mimetype='application/octet-stream')


@app.route("/query", methods=['GET', 'POST'])
def query():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify(msg="Param invalid.", data=[])
        file = request.files['file']
        filename = os.path.join(UPLOAD_DIR, secure_filename(file.filename))
        file.save(filename)
        top_k = int(request.form['n_result'])
        feature = extract_feature(filename)
        scores, names = db.retrieve(feature, top_k)
        return jsonify(msg=f"Success.", data=[*zip(map(str, scores), names)])
