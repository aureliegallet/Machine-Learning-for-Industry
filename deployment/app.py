from flask import Flask, jsonify, render_template
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from pymongo import MongoClient
import torch
import subprocess
from deployment.process_data import DataProcessor

# MongoDB
mongo_client = MongoClient("REMOVED")
db = mongo_client["predictions_db"]
predictions_collection = db["predictions"]
past_data_collection = db["past_data"]

# App
app = Flask(__name__)

data_processor = DataProcessor()

# Function to start TensorBoard
tensorboard_started = False


def start_tensorboard(logdir='../results', port=6006):
    try:
        subprocess.check_call(['pgrep', '-f', 'tensorboard'])
        print("TensorBoard is already running.")
    except subprocess.CalledProcessError:
        print("Starting TensorBoard...")
        subprocess.Popen(['tensorboard', '--logdir', logdir, '--port', str(port)])


@app.before_request
def launch_tensorboard():
    global tensorboard_started
    if not tensorboard_started:
        start_tensorboard()
        tensorboard_started = True


model = torch.load(open("../results/best_model/best_model", "rb"))
model.eval()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


@app.route("/predict")
def predict():
    latest_prediction = predictions_collection.find_one(sort=[("created_at", -1)])

    if latest_prediction:
        result = {
            "id": str(latest_prediction["_id"]),
            "day1_no2": latest_prediction["day1_no2"],
            "day1_o3": latest_prediction["day1_o3"],
            "day2_no2": latest_prediction["day2_no2"],
            "day2_o3": latest_prediction["day2_o3"],
            "day3_no2": latest_prediction["day3_no2"],
            "day3_o3": latest_prediction["day3_o3"],
            "day4_no2": latest_prediction["day4_no2"],
            "day4_o3": latest_prediction["day4_o3"],
            "created_at": latest_prediction["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
            "errors": latest_prediction["errors"],
            "warnings": latest_prediction["warnings"]
        }
        return jsonify(result)
    else:
        return jsonify({"message": "No predictions found."}), 404


@app.route("/previous_prediction")
def previous_prediction():
    previous_prediction_data = predictions_collection.find().sort("created_at", -1).skip(1).limit(1)[0]

    if previous_prediction_data:
        result = {
            "id": str(previous_prediction_data["_id"]),
            "day1_no2": previous_prediction_data["day1_no2"],
            "day1_o3": previous_prediction_data["day1_o3"],
            "day2_no2": previous_prediction_data["day2_no2"],
            "day2_o3": previous_prediction_data["day2_o3"],
            "day3_no2": previous_prediction_data["day3_no2"],
            "day3_o3": previous_prediction_data["day3_o3"],
            "created_at": previous_prediction_data["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
            "errors": previous_prediction_data["errors"],
            "warnings": previous_prediction_data["warnings"]
        }
        return jsonify(result)
    else:
        return jsonify({"message": "No predictions found."}), 404


@app.route("/previous_values")
def previous_values():
    past_data = past_data_collection.find_one(sort=[("created_at", -1)])

    if past_data:
        result = {
            "date_most_recent": past_data['date_most_recent'],
            "no2_most_recent": past_data['no2_most_recent'],
            "o3_most_recent": past_data['o3_most_recent'],
            "date": past_data['date'],
            "no2": past_data['no2'],
            "o3": past_data['o3'],
            "created_at": past_data["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
            "errors": past_data["errors"],
            "warnings": past_data["warnings"]
        }
        return jsonify(result)
    else:
        return jsonify({"message": "No predictions found."}), 404


def update_previous_values():
    past_data, errors, warnings = data_processor.get_past_pollution_data(2, True)

    if len(errors) == 0:
        results = {
            "date_most_recent": past_data.iloc[1]['YYYYMMDD'],
            "no2_most_recent": past_data.iloc[1]['Average_no2'],
            "o3_most_recent": past_data.iloc[1]['Average_o3'],
            "date": past_data.iloc[0]['YYYYMMDD'],
            "no2": past_data.iloc[0]['Average_no2'],
            "o3": past_data.iloc[0]['Average_o3'],
            "created_at": datetime.now(),
            "errors": errors,
            "warnings": warnings
        }

    else:
        results = {
            "date": None,
            "no2": None,
            "o3": None,
            "date_most_recent": None,
            "no2_most_recent": None,
            "o3_most_recent": None,
            "created_at": datetime.now(),
            "errors": errors,
            "warnings": warnings
        }

    past_data_collection.insert_one(results)

    print(f"Previous values updated at {datetime.now()}")


def update_predictions():
    input_data, errors, warnings = get_input()
    if len(errors) != 0 or input_data is None:
        prediction_data = {
            "day1_no2": None,
            "day1_o3": None,
            "day2_no2": None,
            "day2_o3": None,
            "day3_no2": None,
            "day3_o3": None,
            "day4_no2": None,
            "day4_o3": None,
            "created_at": datetime.now(),
            "errors": errors,
            "warnings": warnings
        }

    else:
        with torch.no_grad():
            predictions = model(input_data)
            prediction_values = predictions.numpy().tolist()[0]
            print(prediction_values)

            prediction_data = {
                "day1_no2": prediction_values[0][0][0],
                "day1_o3": prediction_values[0][0][1],
                "day2_no2": prediction_values[0][1][0],
                "day2_o3": prediction_values[0][1][1],
                "day3_no2": prediction_values[0][2][0],
                "day3_o3": prediction_values[0][2][1],
                "day4_no2": prediction_values[0][3][0],
                "day4_o3": prediction_values[0][3][1],
                "created_at": datetime.now(),
                "errors": errors,
                "warnings": warnings
            }

    predictions_collection.insert_one(prediction_data)

    print(f"Predictions updated at {datetime.now()}")


def get_input():
    nn_loader, errors, warnings = data_processor(2)
    return nn_loader, errors, warnings


scheduler = BackgroundScheduler()
# Run the model every morning after the pollution values for the previous day have been recorded
scheduler.add_job(update_predictions, 'cron', hour=6, minute=0)
# Update the previous pollution values every two hours
scheduler.add_job(update_previous_values, 'interval', hours=2)
scheduler.start()
