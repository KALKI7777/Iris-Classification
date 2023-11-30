from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pre-trained machine learning model
model = pickle.load(open('V:\Bharat-Intern\Iris classification\saved_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', result='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input values from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make a prediction using the loaded model
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

        # Pass the result to the template
        return render_template('index.html', result=result)

    except Exception as e:
        # Handle potential errors (e.g., invalid input)
        error_message = f"An error occurred: {e}"
        return render_template('index.html', result=error_message)

if __name__ == '__main__':
    app.run(debug=True)
