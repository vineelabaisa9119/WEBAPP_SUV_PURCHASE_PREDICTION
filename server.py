from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("SUV_Purchase.csv")

app = Flask(__name__)

# deserializing to read the file

model = pickle.load(open("model.pkl", 'rb'))


@app.route('/')
def index():
    return render_template("index.html")  # we are able to render to webpage


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    features = [int(x) for x in request.form.values()]
    print(features)
    final = [np.array(features)]
    x=df.iloc[:,2:4].values
    sst=StandardScaler().fit(x)
    output = model.predict(sst.transform(final))
    print(output)


    if output[0]==0:
        return render_template('index.html', pred=f'The person can not buy SUV' )
    else:
        return render_template('index.html', pred=f'The person can buy SUV')



if __name__ == '__main__':
    app.run(debug=True)

