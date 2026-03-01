import os
import time
import sqlite3
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, Response, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message

from video_processor import VideoProcessor
from auth import hash_password, verify_password, validate_password, generate_otp

load_dotenv()  # Load variables from .env into os.environ

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-me-in-production')
app.config['PERMANENT_SESSION_LIFETIME'] = __import__('datetime').timedelta(days=1)

# Email Configuration (for OTP)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

mail = Mail(app)

# Login Manager Initialization
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'

# Configuration
app.config['VIDEO_PATH'] = Path(app.root_path) / 'static' / 'video' / 'cctv_demo_detection.mp4'
app.config['MODEL_PATH'] = Path(app.root_path) / 'models' / 'best.pt'

def resolve_data_dir() -> Path:
    """Resolve the persistent data directory based on environment."""
    if os.name != 'nt':
        home = os.environ.get('HOME')
        if home:
            # Azure App Service persistent storage mount
            return Path(home) / 'site' / 'data'
    # Fallback to local .data directory for development
    local_data = Path(app.root_path) / '.data'
    return local_data

DATA_DIR = resolve_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / 'database.db'
UPLOADS_DIR = DATA_DIR / 'uploads'
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using persistent data directory: {DATA_DIR}")
print(f"Database path: {DB_PATH}")

def get_db_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def init_db():
    """Ensure database schema and default data exist in persistent storage."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS cameras (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        location TEXT NOT NULL,
        path TEXT NOT NULL
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        is_verified BOOLEAN NOT NULL DEFAULT 0
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS otps (
        email TEXT PRIMARY KEY,
        otp TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        expires_at REAL NOT NULL
    )''')
    
    # Check if we have an admin user
    admin_email = os.environ.get('ADMIN_EMAIL', 'admin@example.com')
    c.execute('SELECT * FROM users WHERE email = ?', (admin_email,))
    if not c.fetchone():
        print(f"Initializing default admin user: {admin_email}")
        admin_pass = os.environ.get('ADMIN_PASSWORD', 'admin123')
        c.execute('INSERT INTO users (email, password_hash, role, is_verified) VALUES (?, ?, ?, 1)',
                  (admin_email, hash_password(admin_pass), 'Admin'))
    
    # Check if we have the default demo camera
    c.execute('SELECT * FROM cameras WHERE name = ?', ('Demo CCTV',))
    if not c.fetchone():
        demo_path = str(Path(app.root_path) / 'static' / 'video' / 'cctv_demo_detection.mp4')
        if Path(demo_path).exists():
            print("Initializing default demo camera")
            c.execute('INSERT INTO cameras (name, type, location, path) VALUES (?, ?, ?, ?)',
                      ('Demo CCTV', 'Video', 'Building A', demo_path))
    
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# Global video processors mapping
processors = {}

@app.route('/health')
def health():
    return {'status': 'ok'}, 200

def get_processor(cam_id):
    if cam_id not in processors:
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM cameras WHERE id = ?', (cam_id,))
        cam = c.fetchone()
        conn.close()
        
        if cam and Path(cam['path']).exists():
             processors[cam_id] = VideoProcessor(
                 cam['path'],
                 model_path=app.config['MODEL_PATH'],
                 on_alert_callback=fire_alert_callback,
                 camera_name=cam['name']
             )
    return processors.get(cam_id)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the home page / login."""
    if current_user.is_authenticated:
        if current_user.role == 'Admin':
            return redirect(url_for('admin'))
        else:
            return redirect(url_for('monitoring'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')

        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        user_row = c.fetchone()
        conn.close()

        if user_row and verify_password(password, user_row['password_hash']):
            if user_row['role'] != role:
                flash(f'Account is not registered as {role}.')
                return redirect(url_for('index'))

            if not user_row['is_verified']:
                flash('Please verify your email first.')
                return redirect(url_for('verify_otp', email=email))

            user = User(user_row['id'], user_row['email'], user_row['role'], user_row['is_verified'])
            login_user(user)
            if user.role == 'Admin':
                return redirect(url_for('admin'))
            return redirect(url_for('monitoring'))
        else:
            flash('Invalid email or password.')

    return render_template('index.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


def send_otp_email(email, otp, subject, purpose_text):
    """Send a branded OTP email."""
    try:
        msg = Message(f'SmartVision FireGuard — {subject}', recipients=[email])
        msg.body = f'Your {purpose_text} code is: {otp}\n\nThis code expires in 10 minutes. Do not share it.'
        msg.html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>SmartVision FireGuard</title></head>
<body style="margin:0;padding:0;background-color:#0b1120;font-family:'Inter',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0b1120;padding:20px 0;">
    <tr><td align="center">
      <table width="420" cellpadding="0" cellspacing="0" style="background-color:#111827;border:1px solid #1f2937;border-radius:10px;overflow:hidden;">
        <tr><td align="center" style="background:linear-gradient(135deg,#1e3a8a,#1e40af);padding:14px 28px;">
          <h1 style="margin:0;color:#fff;font-size:16px;font-weight:700;">🔥 SmartVision FireGuard</h1>
          <p style="margin:2px 0 0;color:#93c5fd;font-size:10px;letter-spacing:.12em;text-transform:uppercase;">AI Fire &amp; Smoke Detection System</p>
        </td></tr>
        <tr><td style="padding:18px 28px;">
          <h2 style="margin:0 0 6px;color:#f9fafb;font-size:14px;font-weight:600;">{subject}</h2>
          <p style="margin:0 0 12px;color:#9ca3af;font-size:12px;line-height:1.5;">Use the code below to proceed. Valid for <strong style="color:#e5e7eb;">10 minutes</strong>.</p>
          <div style="background:#1f2937;border:1px solid #374151;border-radius:8px;padding:12px;text-align:center;margin-bottom:12px;">
            <p style="margin:0 0 2px;color:#6b7280;font-size:10px;letter-spacing:.12em;text-transform:uppercase;">Verification Code</p>
            <p style="margin:0;color:#60a5fa;font-size:30px;font-weight:800;letter-spacing:.3em;font-family:monospace;">{otp}</p>
          </div>
          <p style="margin:0;color:#6b7280;font-size:11px;">If you didn't request this, ignore this email.</p>
        </td></tr>
        <tr><td style="background-color:#0d1424;border-top:1px solid #1f2937;padding:10px 28px;text-align:center;">
          <p style="margin:0;color:#4b5563;font-size:10px;text-transform:uppercase;">© 2026 SmartVision FireGuard</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>'''
        mail.send(msg)
        print(f"Sent OTP {otp} to {email}")
        return True
    except Exception as e:
        print(f"Email failed: {e}. OTP is: {otp}")
        return False


# ── Automatic Fire Alert Emails ────────────────────────────────────

def _get_all_verified_user_emails():
    """Return a list of all verified user email addresses."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT email FROM users WHERE is_verified = 1')
    rows = c.fetchall()
    conn.close()
    return [row['email'] for row in rows]


def send_fire_alert_email(recipients, camera_name):
    """Send a 🚨 FIRE ALERT evacuation email to all verified users."""
    if not recipients:
        return
    try:
        with app.app_context():
            msg = Message(
                '🚨 FIRE ALERT — Immediate Evacuation Required | SmartVision FireGuard',
                recipients=recipients
            )
            msg.html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>FIRE ALERT</title></head>
<body style="margin:0;padding:0;background:#0b1120;font-family:'Inter',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0b1120;padding:20px 0;">
    <tr><td align="center">
      <table width="480" cellpadding="0" cellspacing="0" style="background:#111827;border:2px solid #dc2626;border-radius:12px;overflow:hidden;">
        <tr><td align="center" style="background:linear-gradient(135deg,#7f1d1d,#dc2626);padding:18px 28px;">
          <p style="margin:0 0 4px;color:#fca5a5;font-size:11px;letter-spacing:.2em;text-transform:uppercase;font-weight:700;">⚠️ URGENT ALERT</p>
          <h1 style="margin:0;color:#fff;font-size:22px;font-weight:800;letter-spacing:.02em;">🔥 FIRE DETECTED</h1>
          <p style="margin:4px 0 0;color:#fca5a5;font-size:11px;letter-spacing:.12em;text-transform:uppercase;">SmartVision FireGuard AI System</p>
        </td></tr>
        <tr><td style="padding:24px 28px;">
          <div style="background:#1f2937;border-left:4px solid #dc2626;border-radius:4px;padding:14px 16px;margin-bottom:20px;">
            <p style="margin:0 0 4px;color:#f87171;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;">Detection Source</p>
            <p style="margin:0;color:#f9fafb;font-size:16px;font-weight:700;">{camera_name}</p>
          </div>
          <h2 style="margin:0 0 10px;color:#f9fafb;font-size:16px;font-weight:700;">Immediate Action Required</h2>
          <p style="margin:0 0 16px;color:#d1d5db;font-size:13px;line-height:1.7;">Our AI system has <strong style="color:#f87171;">confirmed a fire</strong> at the monitored location. Please follow your emergency protocol immediately.</p>
          <div style="background:#1f2937;border:1px solid #374151;border-radius:8px;padding:16px;margin-bottom:16px;">
            <p style="margin:0 0 10px;color:#9ca3af;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;">Safety Instructions</p>
            <p style="margin:0 0 8px;color:#e5e7eb;font-size:13px;">🚪 <strong>Evacuate the building immediately</strong> — leave via the nearest emergency exit.</p>
            <p style="margin:0 0 8px;color:#e5e7eb;font-size:13px;">📞 <strong>Call emergency services</strong> (Fire Department: 101) right away.</p>
            <p style="margin:0 0 8px;color:#e5e7eb;font-size:13px;">🚫 <strong>Do NOT use elevators</strong> — use stairwells only.</p>
            <p style="margin:0 0 8px;color:#e5e7eb;font-size:13px;">🧯 <strong>Do NOT re-enter</strong> the building until cleared by fire officials.</p>
            <p style="margin:0;color:#e5e7eb;font-size:13px;">🤝 <strong>Assist others</strong> who may need help evacuating.</p>
          </div>
          <p style="margin:0;color:#6b7280;font-size:11px;">This is an automated alert generated by the SmartVision FireGuard AI system. Do not ignore this message.</p>
        </td></tr>
        <tr><td style="background:#0d1424;border-top:1px solid #374151;padding:12px 28px;text-align:center;">
          <p style="margin:0;color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:.08em;">© 2026 SmartVision FireGuard — Automated Safety Alert</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>'''
            mail.send(msg)
            print(f"[ALERT] Fire alert email sent to {len(recipients)} users for camera: {camera_name}")
    except Exception as e:
        print(f"[ALERT] Failed to send fire alert email: {e}")


def send_all_clear_email(recipients, camera_name):
    """Send a ✅ ALL CLEAR email once fire is no longer detected."""
    if not recipients:
        return
    try:
        with app.app_context():
            msg = Message(
                '✅ ALL CLEAR — Fire Risk Resolved | SmartVision FireGuard',
                recipients=recipients
            )
            msg.html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>All Clear</title></head>
<body style="margin:0;padding:0;background:#0b1120;font-family:'Inter',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0b1120;padding:20px 0;">
    <tr><td align="center">
      <table width="480" cellpadding="0" cellspacing="0" style="background:#111827;border:2px solid #16a34a;border-radius:12px;overflow:hidden;">
        <tr><td align="center" style="background:linear-gradient(135deg,#14532d,#16a34a);padding:18px 28px;">
          <p style="margin:0 0 4px;color:#bbf7d0;font-size:11px;letter-spacing:.2em;text-transform:uppercase;font-weight:700;">✅ SITUATION RESOLVED</p>
          <h1 style="margin:0;color:#fff;font-size:22px;font-weight:800;">ALL CLEAR</h1>
          <p style="margin:4px 0 0;color:#bbf7d0;font-size:11px;letter-spacing:.12em;text-transform:uppercase;">SmartVision FireGuard AI System</p>
        </td></tr>
        <tr><td style="padding:24px 28px;">
          <div style="background:#1f2937;border-left:4px solid #16a34a;border-radius:4px;padding:14px 16px;margin-bottom:20px;">
            <p style="margin:0 0 4px;color:#4ade80;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;">Resolved Source</p>
            <p style="margin:0;color:#f9fafb;font-size:16px;font-weight:700;">{camera_name}</p>
          </div>
          <h2 style="margin:0 0 10px;color:#f9fafb;font-size:16px;font-weight:700;">Fire threat has been resolved</h2>
          <p style="margin:0 0 16px;color:#d1d5db;font-size:13px;line-height:1.7;">The SmartVision FireGuard AI system has confirmed that <strong style="color:#4ade80;">no fire is currently detected</strong> at the monitored location. The situation appears to be under control.</p>
          <div style="background:#1f2937;border:1px solid #374151;border-radius:8px;padding:16px;margin-bottom:16px;">
            <p style="margin:0 0 10px;color:#9ca3af;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;">Next Steps</p>
            <p style="margin:0 0 8px;color:#e5e7eb;font-size:13px;">🚨 Wait for official clearance from fire authorities before re-entering the building.</p>
            <p style="margin:0 0 8px;color:#e5e7eb;font-size:13px;">📋 Report any fire damage to management immediately.</p>
            <p style="margin:0;color:#e5e7eb;font-size:13px;">📱 Continue monitoring via the SmartVision dashboard.</p>
          </div>
          <p style="margin:0;color:#6b7280;font-size:11px;">This is an automated update from the SmartVision FireGuard system.</p>
        </td></tr>
        <tr><td style="background:#0d1424;border-top:1px solid #374151;padding:12px 28px;text-align:center;">
          <p style="margin:0;color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:.08em;">© 2026 SmartVision FireGuard — Automated Safety Alert</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>'''
            mail.send(msg)
            print(f"[CLEAR] All-clear email sent to {len(recipients)} users for camera: {camera_name}")
    except Exception as e:
        print(f"[CLEAR] Failed to send all-clear email: {e}")


def fire_alert_callback(event: str, camera_name: str):
    """Callback wired into every VideoProcessor. Sends alert emails on state transitions."""
    recipients = _get_all_verified_user_emails()
    if event == 'fire_confirmed':
        print(f"[ALERT] Fire CONFIRMED on '{camera_name}' — emailing {len(recipients)} users")
        send_fire_alert_email(recipients, camera_name)
    elif event == 'fire_cleared':
        print(f"[CLEAR] Fire CLEARED on '{camera_name}' — emailing {len(recipients)} users")
        send_all_clear_email(recipients, camera_name)


# ── Forgot Password ────────────────────────────────────────────────
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        user_row = c.fetchone()
        conn.close()
        if user_row:
            otp = generate_otp()
            expires_at = time.time() + 600
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('''INSERT INTO otps (email, otp, password_hash, role, expires_at)
                VALUES (?, ?, '', '', ?)
                ON CONFLICT(email) DO UPDATE SET otp=excluded.otp, expires_at=excluded.expires_at
            ''', (f'reset:{email}', otp, expires_at))
            conn.commit()
            conn.close()
            sent = send_otp_email(email, otp, 'Password Reset', 'password reset')
            if not sent:
                flash(f'Could not send email. OTP (dev only): {otp}')
        flash('If that email is registered, an OTP has been sent.')
        return redirect(url_for('reset_password', email=email))
    return render_template('forgot_password.html')


@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    email = request.args.get('email', '')
    if request.method == 'POST':
        email = request.form.get('email')
        otp_input = request.form.get('otp')
        new_password = request.form.get('new_password')

        if not validate_password(new_password):
            flash('Password must be 8+ chars with uppercase, lowercase, number and special character.')
            return render_template('reset_password.html', email=email)

        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM otps WHERE email = ?', (f'reset:{email}',))
        otp_row = c.fetchone()

        if not otp_row or otp_row['otp'] != otp_input or time.time() > otp_row['expires_at']:
            flash('Invalid or expired OTP. Please try again.')
            conn.close()
            return render_template('reset_password.html', email=email)

        new_hash = hash_password(new_password)
        c.execute('UPDATE users SET password_hash = ? WHERE email = ?', (new_hash, email))
        c.execute('DELETE FROM otps WHERE email = ?', (f'reset:{email}',))
        conn.commit()
        conn.close()
        flash('Password reset successfully! Please log in.')
        return redirect(url_for('index'))
    return render_template('reset_password.html', email=email)


# ── Delete Account ────────────────────────────────────────────────
@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    email = current_user.email
    otp = generate_otp()
    expires_at = time.time() + 600
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO otps (email, otp, password_hash, role, expires_at)
        VALUES (?, ?, '', '', ?)
        ON CONFLICT(email) DO UPDATE SET otp=excluded.otp, expires_at=excluded.expires_at
    ''', (f'delete:{email}', otp, expires_at))
    conn.commit()
    conn.close()
    sent = send_otp_email(email, otp, 'Account Deletion Confirmation', 'account deletion')
    if not sent:
        flash(f'Could not send email. OTP (dev only): {otp}')
    else:
        flash('An OTP has been sent to your email to confirm account deletion.')
    return redirect(url_for('confirm_delete_account'))


@app.route('/confirm_delete_account', methods=['GET', 'POST'])
@login_required
def confirm_delete_account():
    if request.method == 'POST':
        otp_input = request.form.get('otp')
        email = current_user.email
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM otps WHERE email = ?', (f'delete:{email}',))
        otp_row = c.fetchone()

        if not otp_row or otp_row['otp'] != otp_input or time.time() > otp_row['expires_at']:
            flash('Invalid or expired OTP.')
            conn.close()
            return render_template('confirm_delete_account.html')

        c.execute('DELETE FROM users WHERE email = ?', (email,))
        c.execute('DELETE FROM otps WHERE email = ?', (f'delete:{email}',))
        conn.commit()
        conn.close()
        # Send deletion confirmation email
        try:
            from flask_mail import Message as MailMsg
            bye = MailMsg('SmartVision FireGuard — Account Deleted', recipients=[email])
            bye.html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0b1120;font-family:Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0b1120;padding:20px 0;">
    <tr><td align="center">
      <table width="420" cellpadding="0" cellspacing="0" style="background:#111827;border:1px solid #1f2937;border-radius:10px;overflow:hidden;">
        <tr><td align="center" style="background:linear-gradient(135deg,#7f1d1d,#991b1b);padding:16px 28px;">
          <h1 style="margin:0;color:#fff;font-size:17px;font-weight:800;">SmartVision FireGuard</h1>
          <p style="margin:3px 0 0;color:#fca5a5;font-size:10px;letter-spacing:.12em;text-transform:uppercase;">Account Deletion Confirmed</p>
        </td></tr>
        <tr><td style="padding:20px 28px;">
          <h2 style="margin:0 0 8px;color:#f9fafb;font-size:15px;font-weight:700;">Your account has been deleted</h2>
          <p style="margin:0 0 10px;color:#9ca3af;font-size:12px;line-height:1.6;">The account associated with <strong style="color:#e5e7eb;">{email}</strong> has been permanently removed from our systems. All your data has been deleted.</p>
          <p style="margin:0;color:#6b7280;font-size:11px;">If you did not request this deletion, please contact support immediately.</p>
        </td></tr>
        <tr><td style="background:#0d1424;border-top:1px solid #1f2937;padding:10px 28px;text-align:center;">
          <p style="margin:0;color:#4b5563;font-size:10px;text-transform:uppercase;">© 2026 SmartVision FireGuard</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>'''
            mail.send(bye)
            print(f"Deletion confirmation sent to {email}")
        except Exception as e:
            print(f"Deletion email failed: {e}")
        logout_user()
        flash('Your account has been permanently deleted.')
        return redirect(url_for('index'))
    return render_template('confirm_delete_account.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Render the signup page."""
    if current_user.is_authenticated:
        return redirect(url_for('monitoring'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = 'User' # Default new signups to User

        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        if c.fetchone():
            conn.close()
            flash('Email already registered.')
            return redirect(url_for('signup'))
        
        if not validate_password(password):
            conn.close()
            flash('Password must be at least 8 characters, include uppercase, lowercase, number, and special character.')
            return redirect(url_for('signup'))
        
        pass_hash = hash_password(password)
        otp = generate_otp()
        expires_at = time.time() + 600 # 10 mins validity

        # Upsert OTP
        c.execute('''
            INSERT INTO otps (email, otp, password_hash, role, expires_at) 
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET 
            otp=excluded.otp, password_hash=excluded.password_hash, expires_at=excluded.expires_at
        ''', (email, otp, pass_hash, role, expires_at))
        conn.commit()
        conn.close()

        # Try to send email, fallback to printing
        try:
            msg = Message('SmartVision FireGuard — Email Verification', recipients=[email])
            msg.body = f'Your verification code is: {otp}\n\nThis code expires in 10 minutes. Do not share it with anyone.'
            msg.html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SmartVision FireGuard — Email Verification</title>
</head>
<body style="margin:0;padding:0;background-color:#0b1120;font-family:'Inter',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0b1120;padding:20px 0;">
    <tr>
      <td align="center">
        <table width="420" cellpadding="0" cellspacing="0" style="background-color:#111827;border:1px solid #1f2937;border-radius:10px;overflow:hidden;">

          <!-- Header -->
          <tr>
            <td align="center" style="background:linear-gradient(135deg,#1e3a8a,#1e40af);padding:14px 28px;">
              <h1 style="margin:0;color:#ffffff;font-size:16px;font-weight:700;">🔥 SmartVision FireGuard</h1>
              <p style="margin:2px 0 0;color:#93c5fd;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;">AI Fire &amp; Smoke Detection System</p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:18px 28px;">
              <h2 style="margin:0 0 6px;color:#f9fafb;font-size:14px;font-weight:600;">Verify your email</h2>
              <p style="margin:0 0 12px;color:#9ca3af;font-size:12px;line-height:1.5;">
                Enter this code to complete registration. Valid for <strong style="color:#e5e7eb;">10 minutes</strong>.
              </p>

              <!-- OTP Box -->
              <div style="background:#1f2937;border:1px solid #374151;border-radius:8px;padding:12px;text-align:center;margin-bottom:12px;">
                <p style="margin:0 0 2px;color:#6b7280;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;">Verification Code</p>
                <p style="margin:0;color:#60a5fa;font-size:30px;font-weight:800;letter-spacing:0.3em;font-family:monospace;">{otp}</p>
              </div>

              <p style="margin:0;color:#6b7280;font-size:11px;">If you didn't request this, ignore this email.</p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color:#0d1424;border-top:1px solid #1f2937;padding:10px 28px;text-align:center;">
              <p style="margin:0;color:#4b5563;font-size:10px;letter-spacing:0.08em;text-transform:uppercase;">© 2026 SmartVision FireGuard</p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
'''
            mail.send(msg)
            print(f"Sent OTP {otp} to {email}")
        except Exception as e:
            print(f"Failed to send email to {email}: {e}")
            print(f"OTP is: {otp}")
            flash('Could not send email, check terminal for OTP in development.')

        return redirect(url_for('verify_otp', email=email))

    return render_template('signup.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    """Verify OTP and create user account."""
    if current_user.is_authenticated:
        return redirect(url_for('monitoring'))

    email = request.args.get('email') or request.form.get('email')
    
    if request.method == 'POST':
        otp_attempt = request.form.get('otp')
        
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM otps WHERE email = ?', (email,))
        otp_row = c.fetchone()
        
        if not otp_row:
            flash('No pending registration for this email.')
            conn.close()
            return redirect(url_for('signup'))
            
        if time.time() > otp_row['expires_at']:
            c.execute('DELETE FROM otps WHERE email = ?', (email,))
            conn.commit()
            conn.close()
            flash('OTP expired, please sign up again.')
            return redirect(url_for('signup'))
            
        if otp_attempt == otp_row['otp']:
            # Create user
            c.execute('INSERT INTO users (email, password_hash, role, is_verified) VALUES (?, ?, ?, ?)',
                      (email, otp_row['password_hash'], otp_row['role'], True))
            c.execute('DELETE FROM otps WHERE email = ?', (email,))
            conn.commit()
            conn.close()
            # Send welcome email
            try:
                from flask_mail import Message as MailMsg
                welcome = MailMsg('Welcome to SmartVision FireGuard 🔥', recipients=[email])
                welcome.html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0b1120;font-family:Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0b1120;padding:20px 0;">
    <tr><td align="center">
      <table width="420" cellpadding="0" cellspacing="0" style="background:#111827;border:1px solid #1f2937;border-radius:10px;overflow:hidden;">
        <tr><td align="center" style="background:linear-gradient(135deg,#1e3a8a,#1e40af);padding:16px 28px;">
          <h1 style="margin:0;color:#fff;font-size:17px;font-weight:800;">🔥 Welcome to SmartVision FireGuard</h1>
          <p style="margin:3px 0 0;color:#93c5fd;font-size:10px;letter-spacing:.12em;text-transform:uppercase;">AI Fire &amp; Smoke Detection System</p>
        </td></tr>
        <tr><td style="padding:20px 28px;">
          <h2 style="margin:0 0 8px;color:#f9fafb;font-size:15px;font-weight:700;">Your account is ready ✅</h2>
          <p style="margin:0 0 10px;color:#9ca3af;font-size:12px;line-height:1.6;">Hi <strong style="color:#e5e7eb;">{email}</strong>, your account has been successfully created. You can now log in and start monitoring.</p>
          <div style="background:#1f2937;border:1px solid #374151;border-radius:8px;padding:12px;margin-bottom:12px;">
            <p style="margin:0 0 4px;color:#6b7280;font-size:10px;letter-spacing:.1em;text-transform:uppercase;">Registered Email</p>
            <p style="margin:0;color:#60a5fa;font-size:13px;font-weight:700;font-family:monospace;">{email}</p>
          </div>
          <p style="margin:0;color:#6b7280;font-size:11px;">If you didn't create this account, please contact support immediately.</p>
        </td></tr>
        <tr><td style="background:#0d1424;border-top:1px solid #1f2937;padding:10px 28px;text-align:center;">
          <p style="margin:0;color:#4b5563;font-size:10px;text-transform:uppercase;">© 2026 SmartVision FireGuard</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>'''
                mail.send(welcome)
                print(f"Welcome email sent to {email}")
            except Exception as e:
                print(f"Welcome email failed: {e}")
            flash('Email verified! You can now log in.')
            return redirect(url_for('index'))
        else:
            conn.close()
            flash('Incorrect OTP.')
            
    return render_template('verify_otp.html', email=email)

@app.route('/monitoring')
@login_required
def monitoring():
    """Render the monitoring dashboard."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM cameras')
    cameras = c.fetchall()
    conn.close()
    return render_template('monitoring.html', cameras=cameras)

@app.route('/privacy')
def privacy():
    """Render the privacy policy page."""
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    """Render the terms of service page."""
    return render_template('terms.html')

def init_db():
    conn = get_db_connection()
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
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            is_verified BOOLEAN NOT NULL DEFAULT 0
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS otps (
            email TEXT PRIMARY KEY,
            otp TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            expires_at REAL NOT NULL
        )
    ''')
    
    # Create default Admin if no users exist
    c.execute('SELECT COUNT(*) FROM users')
    if c.fetchone()[0] == 0:
        admin_email = os.environ.get('ADMIN_EMAIL')
        admin_password = os.environ.get('ADMIN_PASSWORD')
        
        if admin_email and admin_password:
            admin_pass_hash = hash_password(admin_password)
            c.execute('INSERT INTO users (email, password_hash, role, is_verified) VALUES (?, ?, ?, ?)',
                      (admin_email, admin_pass_hash, 'Admin', True))
            print(f"Created default admin: {admin_email}")
        else:
            print("WARNING: No users found and ADMIN_EMAIL/ADMIN_PASSWORD not set. Admin account NOT created.")

    
    conn.commit()
    conn.close()

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, email, role, is_verified):
        self.id = id
        self.email = email
        self.role = role
        self.is_verified = is_verified

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user_row = c.fetchone()
    conn.close()
    if user_row:
        return User(user_row['id'], user_row['email'], user_row['role'], user_row['is_verified'])
    return None

# Initialize DB on startup
init_db()

@app.route('/admin')
@login_required
def admin():
    if current_user.role != 'Admin':
        return redirect(url_for('monitoring'))
    """Render the admin dashboard with active cameras."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM cameras')
    cameras = c.fetchall()
    conn.close()
    return render_template('admin.html', cameras=cameras)

@app.route('/video_feed/<int:cam_id>')
@login_required
def video_feed(cam_id):
    """Stream video frames."""
    processor = get_processor(cam_id)
    if processor is None:
        return Response(status=404)
    return Response(processor.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    if current_user.role != 'Admin':
        return "Unauthorized", 403
    """Handle video upload and switch monitoring feed."""
    if 'video_file' not in request.files:
        return "No video file provided", 400
    
    file = request.files['video_file']
    if file.filename == '':
        return "No selected file", 400
        
    if file:
        # Save the file securely into the persistent uploads directory
        filename = secure_filename(file.filename)
        upload_path = UPLOADS_DIR / filename
        file.save(upload_path)
        
        # Save to database
        video_name = request.form.get('video_name', filename)
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('INSERT INTO cameras (name, type, location, path) VALUES (?, ?, ?, ?)',
                  (video_name, 'Video', 'Persistent Storage', str(upload_path)))
        cam_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Pre-initialize processor for new video
        processors[cam_id] = VideoProcessor(
            str(upload_path),
            model_path=app.config['MODEL_PATH'],
            on_alert_callback=fire_alert_callback,
            camera_name=video_name
        )
        
        return {"ok": True}, 200
    return {"ok": False}, 400

def _deferred_delete(file_path: Path, delay: float = 3.0):
    """Delete a file after a delay so the video stream has time to close its handle."""
    time.sleep(delay)
    for attempt in range(5):
        try:
            if file_path.exists():
                os.remove(file_path)
            break
        except PermissionError:
            time.sleep(1.0)
        except Exception:
            break

@app.route('/delete/<int:cam_id>', methods=['POST'])
@login_required
def delete_camera(cam_id):
    if current_user.role != 'Admin':
        return "Unauthorized", 403
    """Delete a camera source and its associated physical file."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # 1. Look up the camera to get its path before deleting
    c.execute('SELECT type, path FROM cameras WHERE id = ?', (cam_id,))
    cam = c.fetchone()
    
    if cam:
        # 2. Delete the record from the database
        c.execute('DELETE FROM cameras WHERE id = ?', (cam_id,))
        conn.commit()
        
        # 3. If it was an uploaded video, stop processor and schedule file deletion
        if cam['type'] == 'Video':
            file_path = Path(cam['path'])

            # Stop processor — signals the stream generator to exit and releases self.cap
            if cam_id in processors:
                try:
                    processors[cam_id].stop()
                    del processors[cam_id]
                except Exception as e:
                    print(f"Error stopping processor: {e}")

            # Delete in a background thread after a delay so the streaming
            # HTTP connection has time to fully close and release the file handle
            if file_path.name != 'cctv_demo_detection.mp4':
                t = threading.Thread(target=_deferred_delete, args=(file_path,), daemon=True)
                t.start()
                
    conn.close()
    return redirect(url_for('admin'))

@app.route('/uploads/<path:filename>')
@login_required
def serve_upload(filename):
    """Serve uploaded video files from the persistent storage directory."""
    return send_from_directory(str(UPLOADS_DIR), filename)

@app.route('/api/cameras')
@login_required
def get_cameras():
    """Return a list of current cameras for dynamic frontend polling."""
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT id, name, location FROM cameras')
    cameras = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"cameras": cameras}

@app.route('/api/stats/<int:cam_id>')
@login_required
def get_stats(cam_id):
    """Return current inference statistics."""
    processor = get_processor(cam_id)
    if processor is None:
        return {"metrics": None, "logs": [], "growth_history": []}
    return processor.get_stats()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)

