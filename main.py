# main.py

from app import app
from app import server

# Import the callbacks to ensure they are registered
import callbacks

if __name__ == '__main__':
    app.run_server(debug=True)