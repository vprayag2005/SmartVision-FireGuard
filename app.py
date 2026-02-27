import os
import time
from pathlib import Path

from flask import Flask, render_template, Response, request, redirect, url_for
from werkzeug.utils import secure_filename

from video_processor import VideoProcessor

app = Flask(__name__)

# Configuration
app.config['VIDEO_PATH'] = Path(app.root_path) / 'static' / 'video' / 'cctv_demo_detection.mp4'
app.config['MODEL_PATH'] = Path(app.root_path) / 'models' / 'best.pt'

# Global video processors mapping
processors = {}

def get_processor(cam_id):
    if cam_id not in processors:
        conn = sqlite3.connect('database.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM cameras WHERE id = ?', (cam_id,))
        cam = c.fetchone()
        conn.close()
        
        if cam and Path(cam['path']).exists():
             processors[cam_id] = VideoProcessor(cam['path'], model_path=app.config['MODEL_PATH'])
    return processors.get(cam_id)

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/signup')
def signup():
    """Render the signup page."""
    return render_template('signup.html')

@app.route('/monitoring')
def monitoring():
    """Render the monitoring dashboard."""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM cameras')
    cameras = c.fetchall()
    conn.close()
    return render_template('monitoring.html', cameras=cameras)

import sqlite3

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            location TEXT NOT NULL,
            path TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

@app.route('/admin')
def admin():
    """Render the admin dashboard with active cameras."""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM cameras')
    cameras = c.fetchall()
    conn.close()
    return render_template('admin.html', cameras=cameras)

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    """Stream video frames."""
    processor = get_processor(cam_id)
    if processor is None:
        return Response(status=404)
    return Response(processor.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and switch monitoring feed."""
    if 'video_file' not in request.files:
        return "No video file provided", 400
    
    file = request.files['video_file']
    if file.filename == '':
        return "No selected file", 400
        
    if file:
        # Save the file securely into the 'uploads' subdirectory
        filename = secure_filename(file.filename)
        uploads_dir = Path(app.root_path) / 'static' / 'video' / 'uploads'
        uploads_dir.mkdir(exist_ok=True, parents=True)
        
        upload_path = uploads_dir / filename
        file.save(upload_path)
        
        # Save to database
        video_name = request.form.get('video_name', filename)
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('INSERT INTO cameras (name, type, location, path) VALUES (?, ?, ?, ?)',
                  (video_name, 'Video', upload_path.name, str(upload_path)))
        cam_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Pre-initialize processor for new video
        processors[cam_id] = VideoProcessor(
            str(upload_path), 
            model_path=app.config['MODEL_PATH']
        )
        
        return redirect(url_for('admin'))
    return "Failed to upload video", 400

@app.route('/delete/<int:cam_id>', methods=['POST'])
def delete_camera(cam_id):
    """Delete a camera source and its associated physical file."""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # 1. Look up the camera to get its path before deleting
    c.execute('SELECT type, path FROM cameras WHERE id = ?', (cam_id,))
    cam = c.fetchone()
    
    if cam:
        # 2. Delete the record from the database
        c.execute('DELETE FROM cameras WHERE id = ?', (cam_id,))
        conn.commit()
        
        # 3. If it was an uploaded video and it exists on disk, delete the physical file
        if cam['type'] == 'Video':
            file_path = Path(cam['path'])
            
            # 4. Remove from running processor memory and STOP it first
            if cam_id in processors:
                try:
                    processors[cam_id].stop()
                    del processors[cam_id]
                except Exception as e:
                    print(f"Error stopping processor: {e}")
            
            # Give Windows a split second to release the file handle
            time.sleep(0.5)
            
            print(f"[DELETE] Attempting to remove: {file_path}")
            print(f"[DELETE] File exists: {file_path.exists()}")
            try:
                if file_path.exists() and file_path.name != 'cctv_demo_detection.mp4':
                    os.remove(file_path)
                    print(f"[DELETE] Successfully deleted: {file_path}")
                else:
                    print(f"[DELETE] Skipped (not found or protected file)")
            except Exception as e:
                print(f"[DELETE] Error removing file: {e}")
                
    conn.close()
    return redirect(url_for('admin'))

@app.route('/api/cameras')
def get_cameras():
    """Return a list of current cameras for dynamic frontend polling."""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT id, name, location FROM cameras')
    cameras = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"cameras": cameras}

@app.route('/api/stats/<int:cam_id>')
def get_stats(cam_id):
    """Return current inference statistics."""
    processor = get_processor(cam_id)
    if processor is None:
        return {"metrics": None, "logs": [], "growth_history": []}
    return processor.get_stats()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
