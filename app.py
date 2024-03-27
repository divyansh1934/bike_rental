from flask import Flask,request,render_template

from src.BikeSharePrediction.pipelines.prediction_pipeline import PredictPipeline,CustomData

app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    else:
        data=CustomData(
            season=float(request.form.get("season")),
            yr=float(request.form.get("yr")),
            mnth=float(request.form.get("mnth")),
            holiday=float(request.form.get("holiday")),
            weekday=float(request.form.get("weekday")),
            workingday=float(request.form.get("workingday")),
            weathersit=request.form.get("weathersit"),
            temp=request.form.get("temp"),
            atemp=request.form.get("atemp"),
            hum=request.form.get("hum"),
            windspeed=request.form.get("windspeed")
        )
        final_data=data.get_data_as_dataframe()

        predict_pipeline=PredictPipeline()

        pred=predict_pipeline.predict(final_data)

        result=round(pred[0],2)

        return render_template("result.html",final_result=result)



if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)