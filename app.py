from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify
)
import pandas as pd
import os
from werkzeug.utils import secure_filename
from model import MLModel

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load or initialize model
try:
    ml_model = MLModel.load('models/trained_model.pkl')
except:
    ml_model = MLModel()
    print("No pre-trained model found. Using new model instance.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read CSV
            df = pd.read_csv(filepath)
            
            # Assume last column is the target variable
            # Modify this based on your specific use case
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Evaluate the model
            metrics, predictions = ml_model.evaluate(X, y)
            
            # Format metrics for display
            metrics_formatted = {
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}"
            }
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template(
                'results.html',
                metrics=metrics_formatted,
                sample_size=len(df)
            )
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a CSV file.')
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train_model():
    """Optional endpoint to train/retrain the model"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            ml_model.train(X, y)
            ml_model.save('models/trained_model.pkl')
            
            os.remove(filepath)
            
            return jsonify({'message': 'Model trained successfully'}), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True)