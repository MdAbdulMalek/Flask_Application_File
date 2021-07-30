from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
import os
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from utils import *

app = Flask(__name__)
app.secret_key = "super secret key"

client_filename = ""
sanveo_filename = ""
output_filename = ""

app.config["DEBUG"] = True



UPLOAD_FOLDER = "static/files"

OUTPUT_FOLDER = "static/outputs"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html", flag=0)


## Function for uploading sanveo file
@app.route("/upload_client", methods=["POST"])
def upload_file_client():
    global client_filename
	# check if the post request has the file part
    if 'source_fileName' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
	
    files = request.files.getlist('source_fileName')
	
    errors = {}
    success = False
	
    for file in files:
        if file and allowed_file(file.filename):
            client_filename = "client.csv"
            path = os.path.join(app.config['UPLOAD_FOLDER'], client_filename)
            if os.path.exists(path):
                os.remove(path)
            file.save(path)
            success = True
            
        else:
            errors[file.filename] = 'File type is not allowed'

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 206
        return resp
    if success:
        resp = jsonify({'message' : 'Files successfully uploaded'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp


## Function for uploading savneo file
@app.route("/upload_sanveo", methods=["GET", "POST"])
def upload_file_sanveo():
    global sanveo_filename


	# check if the post request has the file part
    if 'source_fileName_Sanveo' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
	
    files = request.files.getlist('source_fileName_Sanveo')
	
    errors = {}
    success = False
	
    for file in files:
        if file and allowed_file(file.filename):
            sanveo_filename = "sanveo.csv"
            path = os.path.join(app.config['UPLOAD_FOLDER'], sanveo_filename)
            if os.path.exists(path):
                os.remove(path)

            file.save(path)
            success = True
            
        else:
            errors[file.filename] = 'File type is not allowed'

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 206
        return resp
    if success:
        resp = jsonify({'message' : 'Files successfully uploaded'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp



## Main processing function
@app.route("/process", methods=["GET","POST"])
def process_match():

    global client_filename
    global output_filename

    check_msg= ""

    resp = {}
    ## Processing client and sanveo file
    client_file = os.path.join(UPLOAD_FOLDER, client_filename)
    df_client = pd.read_csv(client_file)

    sanveo_file = os.path.join(UPLOAD_FOLDER, sanveo_filename)
    df_sanveo = pd.read_csv(sanveo_file)

    

    

    tmp_cols = request.form.getlist('sourceHeaderFieldsClient')
    selected_cols = tmp_cols[0].split(',')


    list_check = False
    print(f"Sanveo Availabe Columns \n\n")
    print(df_sanveo.columns)

    print(f"Client Availabe Columns\n\n")
    print(df_client.columns)
    for s in selected_cols:
        if s not in df_sanveo.columns:
            print(f"{s} not found!")
            list_check = True
    if list_check:
        check_msg = "Some column names didn't match"
        resp['message_match'] = check_msg
        resp['flag'] = 0
        return resp

    else:
        check_msg = "All column names matched"

    
    df_client_work = df_client[selected_cols].copy(deep=True)

    
    df_sanveo_work = df_sanveo[selected_cols].copy(deep=True)


   


    if "Size" in selected_cols:
        df_client_work["Size"] = df_client["Size"].map(lambda x: strip_string(x))
        df_sanveo_work["Size"] = df_sanveo_work["Size"].map(lambda x: strip_string(x))

    df_client_work = df_client_work.applymap(lambda x: str(x).lower())
    df_sanveo_work = df_sanveo_work.applymap(lambda x: str(x).lower())
    client_feat = list(df_client_work.agg(" ".join, axis=1))
    cat_feat = list(df_sanveo_work.agg(" ".join, axis=1))
    cat_lab = list(df_sanveo["ID"].values)

    cat_feat_ind_dict = dict((key, val) for val, key in enumerate(cat_feat))

    feat_2_label_dict = {}
    label_2_feat_dict = {}

    for i, f in enumerate(cat_feat):
        feat_2_label_dict[f] = cat_lab[i]
        label_2_feat_dict[cat_lab[i]] = f

    result = []
    conf_list = []

    for ind, sample in enumerate(client_feat):
        sample_number = "sample_number_" + str(ind + 1)
        findings = process.extract(sample, cat_feat, scorer=fuzz.ratio, limit=1)
        pred_labels, pred_confs, features = find_labels(
            cat_feat_ind_dict, cat_lab, findings
        )
        label = pred_labels[0]
        conf = pred_confs[0]
        conf_list.append(conf)
        if conf == 100:
            result.append(label)
        elif conf > 85 and conf < 100:
            findings = process.extract(sample, cat_feat, scorer=fuzz.ratio, limit=3)
            pred_labels, _, _ = find_labels(cat_feat_ind_dict, cat_lab, findings)
            result.append(pred_labels)
        else:
            findings = process.extract(sample, cat_feat, scorer=fuzz.ratio, limit=5)
            pred_labels, _, _ = find_labels(cat_feat_ind_dict, cat_lab, findings)
            result.append(pred_labels)

    df_client_work["Predicted_label"] = result
    df_client_work["Confidence"] = conf_list

    output_filename = client_filename.split('.')[0] + "_" + "output.csv"
    if os.path.exists(os.path.join(OUTPUT_FOLDER, output_filename)):
        os.remove(os.path.join(OUTPUT_FOLDER, output_filename))


    df_client_work.to_csv(os.path.join(OUTPUT_FOLDER, output_filename))

    msg = "Processing Finished"
    down_msg = "Download the output file"

    resp['message_match'] = check_msg
    resp['message_finish'] = msg
    resp['flag'] = 1
    resp['download_msg'] = down_msg

    return resp


@app.route('/download')
def download_output_file():
    global output_filename
    path = os.path.join(OUTPUT_FOLDER, output_filename)
    return send_file(path, as_attachment=True)



if __name__ == "__main__":
    app.run(host="0.0.0.0")
