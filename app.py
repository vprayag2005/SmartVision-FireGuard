from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/signup')
@app.route('/signup.html')
def signup():
    return render_template('signup.html')

@app.route('/monitoring')
@app.route('/monitoring.html')
def monitoring():
    return render_template('monitoring.html')

@app.route('/admin')
@app.route('/admin.html')
def admin():
    return render_template('admin.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
