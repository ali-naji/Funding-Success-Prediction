from flask import Blueprint, request, render_template, redirect, flash, jsonify
from funding_model.predict import predict
import logging

logger = logging.getLogger(__name__)
prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/main')
def main():
    return redirect('/')


@prediction_app.route('/about')
def about():
    return render_template('about.html')


@prediction_app.route('/')
def index():
    return render_template('home.html')


@prediction_app.route('/contact_us', methods=['GET', 'POST'])
def contact_us():
    if request.method == 'GET':
        return render_template('contact_us.html')
    else:
        flash("Message Sent Successfully. Thank you for your feedback")
        return redirect('/contact_us')


@prediction_app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'GET':
        return render_template('form.html')
    elif request.method == 'POST':
        input_data = {key: [val] for (key, val) in request.form.items()}
        input_data = order_input_dict(input_data)
        logger.info(f'inputs : {input_data}')
        result = predict(input_data)
        logger.info(f'result : {result}')
        return render_template('result.html', prediction=result['predictions'][0])


@prediction_app.route('/test_prediction', methods=['POST'])
def test_prediction():
    json_data = request.get_json()
    logger.debug(f'Inputs: {json_data}')

    result = predict(json_data)
    predictions = result.get('predictions').tolist()
    version = result.get('version')

    return jsonify({'predictions': predictions,
                    'version': version})


def order_input_dict(input_dict):
    return {'desc': input_dict['desc'], 'goal': [float(input_dict['goal'][0])], 'keywords': input_dict['keywords'], 'disable_communication': [int(input_dict['disable_communication'][0])], 'country': input_dict['country'], 'currency': input_dict['currency'], 'backers_count': [float(input_dict['backers_count'][0])]}
