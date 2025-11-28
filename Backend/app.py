from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app) # Enable CORS for all routes (adjust as needed for production)

# Store your n8n Webhook Production URL securely as an environment variable
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "YOUR_N8N_WEBHOOK_URL")

@app.route("/api/submit-data", methods=["POST"])
def submit_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Forward the data to the n8n webhook URL
        n8n_response = requests.post(N8N_WEBHOOK_URL, json=data)
        n8n_response.raise_for_status() # Raise an exception for bad status codes

        return jsonify({"message": "Data sent to n8n successfully", "n8n_status": n8n_response.status_code}), 200

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to send data to n8n: {str(e)}"}), 500

@app.route("/dashboard", methods=["GET"])
def get_dashboard_data():
    # Get stock data from yfinance, database, etc.

@app.route("/api/stock/<symbol>", methods=["GET"])
def get_stock_analysis(symbol):
    try:
        stock = Stock.query.filter_by(symbol=symbol.upper()).first()
        if not stock:
            return jsonify({"error": "Stock not found"}), 404

        # Retrieve news, trends, and LLM data
        news_data = News.query.filter_by(symbol=symbol.upper()).order_by(News.date.desc()).limit(10).all()
        # Similarly, query for trends and LLM data if you store them

        response_data = stock.to_dict()
        response_data['news'] = [article.to_dict() for article in news_data]
        # Add trends and LLM data to response_data

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)