from flask import Flask, render_template, request, jsonify
from mental_health_detector import MentalHealthDetector
import os

app = Flask(__name__)
detector = MentalHealthDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        analysis = detector.analyze_text(text)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 