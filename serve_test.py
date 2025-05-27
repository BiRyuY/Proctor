import http.server
import socketserver
import os
import webbrowser   
from pathlib import Path

# Define the port
PORT = 8000

# Function to find an available port
def find_available_port(start_port):
    port = start_port
    max_port = start_port + 100  # Try up to 100 ports
    
    while port < max_port:
        try:
            with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as test_server:
                test_server.server_close()
                return port
        except OSError:
            port += 1
    
    raise RuntimeError("Could not find an available port")

# Find an available port
PORT = find_available_port(PORT)

# Create a simple HTTP server
Handler = http.server.SimpleHTTPRequestHandler

# Ensure the test_page.html file exists
html_path = Path("test_page.html")
if not html_path.exists():
    print(f"Error: {html_path} not found!")
    exit(1)

print(f"Starting server at http://localhost:{PORT}")
print(f"Test page available at http://localhost:{PORT}/test_page.html")

# Start the server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Server started. Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
