import os
from pathlib import Path

from flask import Flask, render_template, Response

from video_processor import VideoProcessor

app = Flask(__name__)
VIDEO_PATH = Path(app.root_path) / 'static' / 'video' / 'cctv_demo_detection.mp4'
MODEL_PATH = Path(app.root_path) / 'models' / 'best.pt'

# Global video processor instance
processor = VideoProcessor(VIDEO_PATH, model_path=MODEL_PATH)

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

@app.route('/monitoring-video')
@app.route('/monitoring-video.html')
@app.route('/monitoring_video.html')
def monitoring_video():
    return render_template('monitoring_video.html')


@app.route('/video_feed')
def video_feed():
    return Response(processor.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    return processor.get_stats()



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

