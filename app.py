from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/monitoring')
@app.route('/monitoring.html')
def monitoring():
    return render_template('monitoring.html')

@app.route('/admin')
@app.route('/admin.html')
def admin():
    return render_template('admin.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
