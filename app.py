from flask import Flask, request, render_template
from src.BikeSharePrediction.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.BikeSharePrediction.logger.logging import log_info

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return render_template('predicted_demand.html', demand=None)
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    custom_data = CustomData(
        season=int(request.form.get('season')),
        yr=int(request.form.get('yr')),
        mnth=int(request.form.get('mnth')),
        holiday=int(request.form.get('holiday')),
        weekday=int(request.form.get('weekday')),
        workingday=int(request.form.get('workingday')),
        weathersit=int(request.form.get('weathersit')),
        temp=float(request.form.get('temp')),
        atemp=float(request.form.get('atemp')),
        hum=float(request.form.get('hum')),
        windspeed=float(request.form.get('windspeed'))
    )
    final_data = custom_data.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(final_data)

    log_info('Dataframe Gathered')

    return render_template("predicted_demand.html", demand=round(pred[0], 0))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
