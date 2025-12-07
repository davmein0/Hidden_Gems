from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (adjust as needed for production)

N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "YOUR_N8N_WEBHOOK_URL")


@app.route("/api/submit-data", methods=["POST"])
def submit_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        n8n_response = requests.post(N8N_WEBHOOK_URL, json=data)
        n8n_response.raise_for_status()
        return jsonify({"message": "Data sent to n8n successfully", "n8n_status": n8n_response.status_code}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to send data to n8n: {str(e)}"}), 500


@app.route("/dashboard", methods=["GET"])
def get_dashboard_data():
    # TODO: Implement data retrieval for dashboard (yfinance, database, etc.)
    return jsonify({"message": "Not implemented"}), 501


@app.route("/api/stock/<symbol>", methods=["GET"])
def get_stock_analysis(symbol):
    try:
        # TODO: Wire to real DB. This is a placeholder using models.
        from .models import Stock, News

        stock = Stock.query.filter_by(symbol=symbol.upper()).first()
        if not stock:
            return jsonify({"error": "Stock not found"}), 404

        news_data = News.query.filter_by(symbol=symbol.upper()).order_by(News.date.desc()).limit(10).all()
        response_data = stock.to_dict()
        response_data["news"] = [article.to_dict() for article in news_data]
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
