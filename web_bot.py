from flask import Flask, render_template_string
import threading
from PyQt5.QtWidgets import QApplication
import sys

app = Flask(__name__)

# Your existing bot GUI wrapped
def run_bot():
    app = QApplication(sys.argv)
    ex = EnhancedTradingBotGUI()
    sys.exit(app.exec_())

# Simple web interface
@app.route('/')
def home():
    return render_template_string('''
        <h1>Trading Bot Dashboard</h1>
        <iframe src="/bot" width="1000" height="600"></iframe>
    ''')

@app.route('/bot')
def bot_interface():
    return render_template_string('''
        <h2>Bot is running in background</h2>
        <p>Check logs below:</p>
        <iframe src="/log" width="800" height="400"></iframe>
    ''')

@app.route('/log')
def show_log():
    with open('trading_bot_debug.log') as f:
        logs = f.read()
    return render_template_string('<pre>{{ logs }}</pre>', logs=logs)

if __name__ == '__main__':
    # Start bot in background thread
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Start web server
    app.run(host='0.0.0.0', port=5000)