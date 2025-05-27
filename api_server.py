from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store test data
test_data = {
    "tab_switches": 0,
    "fullscreen_exits": 0,
    "duration": 0,
    "test_completed": False
}

@app.route('/api/test-data', methods=['GET'])
def get_test_data():
    logger.info(f"GET /api/test-data - Returning: {test_data}")
    return jsonify(test_data)

@app.route('/api/test-data', methods=['POST'])
def update_test_data():
    global test_data
    try:
        data = request.json
        logger.info(f"POST /api/test-data - Received: {data}")
        test_data.update(data)
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error in update_test_data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/test-complete', methods=['POST'])
def complete_test():
    global test_data
    try:
        data = request.json
        logger.info(f"POST /api/test-complete - Received: {data}")
        test_data.update(data)
        test_data["test_completed"] = True
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error in complete_test: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/ping', methods=['GET'])
def ping():
    """Simple endpoint to test if the server is running"""
    logger.info("Ping received")
    return jsonify({"status": "ok", "message": "API server is running"})

def start_api_server(port=5000):
    """Start the Flask API server in a separate thread"""
    def run_app():
        try:
            logger.info(f"Starting API server on port {port}")
            app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
    
    thread = threading.Thread(target=run_app)
    thread.daemon = True  # Make thread exit when main thread exits
    thread.start()
    
    # Wait a moment to ensure server is up
    time.sleep(1)
    
    # Log a test message
    logger.info(f"API server started on port {port}")
    logger.info(f"Initial test_data: {test_data}")
    
    return port

if __name__ == "__main__":
    port = start_api_server()
    logger.info(f"API server running on port {port}. Press Ctrl+C to exit.")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")


