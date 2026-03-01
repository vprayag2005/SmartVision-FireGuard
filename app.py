import os
import shutil
import secrets
import time
import sqlite3
import threading
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load variables from .env into os.environ
except ImportError:
    pass  # Azure sets environment variables natively

from flask import Flask, render_template, Response, request, redirect, url_for, flash, send_from_directory, session
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message

from video_processor import VideoProcessor
from auth import hash_password, verify_password, validate_password, generate_otp

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
        # Azure App Service persistent storage is mounted under /home.
        candidates = [Path('/home/site/data')]
        home = os.environ.get('HOME')
        if home:
            candidates.append(Path(home) / 'site' / 'data')
        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            except OSError:
                continue
    # Fallback to local .data directory for development
    return Path(app.root_path) / '.data'


def migrate_legacy_data(target_dir: Path) -> None:
    """One-time migration from old ephemeral root path to persistent Azure path."""
    if os.name == 'nt':
        return
    target_db = target_dir / 'database.db'
    if target_db.exists():
        return

    legacy_candidates = [Path('/root/site/data')]
    home = os.environ.get('HOME')
    if home:
        legacy_candidates.append(Path(home) / 'site' / 'data')

    seen = set()
    for legacy_dir in legacy_candidates:
        legacy_str = str(legacy_dir)
        if legacy_str in seen:
            continue
        seen.add(legacy_str)

        if legacy_dir == target_dir:
            continue
        legacy_db = legacy_dir / 'database.db'
        if not legacy_db.exists():
            continue

        try:
            print(f"Migrating data directory from {legacy_dir} to {target_dir}")
            shutil.copy2(legacy_db, target_db)
            legacy_uploads = legacy_dir / 'uploads'
            target_uploads = target_dir / 'uploads'
            if legacy_uploads.exists():
                shutil.copytree(legacy_uploads, target_uploads, dirs_exist_ok=True)
            print("Legacy data migration completed")
            return
        except Exception as e:
            print(f"Legacy data migration failed from {legacy_dir}: {e}")

DATA_DIR = resolve_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / 'database.db'
UPLOADS_DIR = DATA_DIR / 'uploads'
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
migrate_legacy_data(DATA_DIR)

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
processor_errors = {}
processors_initializing = set()
processors_lock = threading.Lock()

# Active monitoring client limiter
MONITORING_MAX_CLIENTS = int(os.environ.get('MONITORING_MAX_CLIENTS', '10'))
MONITORING_CLIENT_TTL_SECONDS = int(os.environ.get('MONITORING_CLIENT_TTL_SECONDS', '180'))
monitoring_clients = {}
monitoring_clients_lock = threading.Lock()


def _prune_monitoring_clients_locked(now_ts: float) -> None:
    stale_cutoff = now_ts - MONITORING_CLIENT_TTL_SECONDS
    stale_ids = [cid for cid, seen_at in monitoring_clients.items() if seen_at < stale_cutoff]
    for cid in stale_ids:
        monitoring_clients.pop(cid, None)


def _register_monitoring_client(client_id: str) -> tuple[bool, int]:
    now_ts = time.time()
    with monitoring_clients_lock:
        _prune_monitoring_clients_locked(now_ts)
        if client_id in monitoring_clients:
            monitoring_clients[client_id] = now_ts
            return True, len(monitoring_clients)
        if len(monitoring_clients) >= MONITORING_MAX_CLIENTS:
            return False, len(monitoring_clients)
        monitoring_clients[client_id] = now_ts
        return True, len(monitoring_clients)


def _touch_monitoring_client(client_id: str) -> tuple[bool, int]:
    now_ts = time.time()
    with monitoring_clients_lock:
        _prune_monitoring_clients_locked(now_ts)
        if client_id in monitoring_clients:
            monitoring_clients[client_id] = now_ts
            return True, len(monitoring_clients)
        if len(monitoring_clients) >= MONITORING_MAX_CLIENTS:
            return False, len(monitoring_clients)
        monitoring_clients[client_id] = now_ts
        return True, len(monitoring_clients)


def _release_monitoring_client(client_id: str | None) -> int:
    if not client_id:
        return _active_monitoring_client_count()
    with monitoring_clients_lock:
        monitoring_clients.pop(client_id, None)
        _prune_monitoring_clients_locked(time.time())
        return len(monitoring_clients)


def _active_monitoring_client_count() -> int:
    with monitoring_clients_lock:
        _prune_monitoring_clients_locked(time.time())
        return len(monitoring_clients)





@app.route('/health')
def health():
    return {'status': 'ok'}, 200

def _init_processor(cam_id):
    """Create a processor in a background thread and record any initialization error."""
    try:
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM cameras WHERE id = ?', (cam_id,))
        cam = c.fetchone()
        conn.close()

        if not cam:
            raise RuntimeError(f"Camera {cam_id} not found in database")

        video_path = Path(cam['path'])
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found for camera {cam_id}: {video_path}")

        processor = VideoProcessor(
            str(video_path),
            model_path=app.config['MODEL_PATH'],
            on_alert_callback=fire_alert_callback,
            camera_name=cam['name']
        )
        with processors_lock:
            processors[cam_id] = processor
            processor_errors.pop(cam_id, None)
    except Exception as e:
        with processors_lock:
            processor_errors[cam_id] = str(e)
        print(f"Processor initialization failed for cam {cam_id}: {e}")
    finally:
        with processors_lock:
            processors_initializing.discard(cam_id)


def _ensure_processor_initializing(cam_id):
    """Start asynchronous initialization once. Returns current processor (if already ready)."""
    with processors_lock:
        existing = processors.get(cam_id)
        if existing is not None:
            return existing
        if cam_id in processors_initializing:
            return None
        processors_initializing.add(cam_id)
        processor_errors.pop(cam_id, None)

    threading.Thread(target=_init_processor, args=(cam_id,), daemon=True).start()
    return None


def get_processor(cam_id, wait=False, wait_timeout=40.0):
    """Return processor; optionally wait for async initialization (used by video feed)."""
    processor = _ensure_processor_initializing(cam_id)
    if processor is not None or not wait:
        return processor

    deadline = time.time() + wait_timeout
    while time.time() < deadline:
        with processors_lock:
            ready = processors.get(cam_id)
            still_initializing = cam_id in processors_initializing
        if ready is not None:
            return ready
        if not still_initializing:
            break
        time.sleep(0.05)
    return None




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
    monitoring_client_id = session.pop('monitoring_client_id', None)
    _release_monitoring_client(monitoring_client_id)
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
    previous_client_id = session.pop('monitoring_client_id', None)
    _release_monitoring_client(previous_client_id)

    monitoring_client_id = secrets.token_urlsafe(18)
    allowed, active_clients = _register_monitoring_client(monitoring_client_id)
    if not allowed:
        return redirect(url_for('server_busy', active=active_clients, max_clients=MONITORING_MAX_CLIENTS))
    session['monitoring_client_id'] = monitoring_client_id

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM cameras')
    cameras = c.fetchall()
    conn.close()
    return render_template(
        'monitoring.html',
        cameras=cameras,
        monitoring_client_id=monitoring_client_id
    )


@app.route('/server-busy')
def server_busy():
    active_clients = request.args.get('active', type=int)
    max_clients = request.args.get('max_clients', type=int) or MONITORING_MAX_CLIENTS
    if active_clients is None:
        active_clients = _active_monitoring_client_count()
    return render_template(
        'server_busy.html',
        active_clients=active_clients,
        max_clients=max_clients
    ), 503


@app.route('/api/monitoring/heartbeat', methods=['POST'])
@login_required
def monitoring_heartbeat():
    data = request.get_json(silent=True) or {}
    client_id = data.get('client_id') or session.get('monitoring_client_id')
    session_client_id = session.get('monitoring_client_id')
    if not client_id:
        return {"ok": False, "reason": "missing_client_id"}, 400
    if session_client_id and client_id != session_client_id:
        return {"ok": False, "reason": "invalid_client_id"}, 403

    allowed, active_clients = _touch_monitoring_client(client_id)
    if not allowed:
        return {
            "ok": False,
            "reason": "capacity_reached",
            "active_clients": active_clients,
            "max_clients": MONITORING_MAX_CLIENTS
        }, 429
    return {
        "ok": True,
        "active_clients": active_clients,
        "max_clients": MONITORING_MAX_CLIENTS
    }


@app.route('/api/monitoring/release', methods=['POST'])
@login_required
def monitoring_release():
    data = request.get_json(silent=True) or {}
    client_id = data.get('client_id') or session.get('monitoring_client_id')
    session_client_id = session.get('monitoring_client_id')
    if session_client_id and client_id == session_client_id:
        session.pop('monitoring_client_id', None)
    active_clients = _release_monitoring_client(client_id)
    return {"ok": True, "active_clients": active_clients}

@app.route('/privacy')
def privacy():
    """Render the privacy policy page."""
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    """Render the terms of service page."""
    return render_template('terms.html')

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
    processor = get_processor(cam_id, wait=True, wait_timeout=45.0)
    if processor is None:
        with processors_lock:
            error_message = processor_errors.get(cam_id)
        if error_message:
            return Response(f"Unable to initialize video stream: {error_message}", status=500)
        return Response("Video stream is initializing. Please retry.", status=503)
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
        location = request.form.get('location', '').strip() or 'Unspecified'
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('INSERT INTO cameras (name, type, location, path) VALUES (?, ?, ?, ?)',
                  (video_name, 'Video', location, str(upload_path)))
        cam_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Pre-initialize processor for new video
        preloaded_processor = VideoProcessor(
            str(upload_path),
            model_path=app.config['MODEL_PATH'],
            on_alert_callback=fire_alert_callback,
            camera_name=video_name
        )
        with processors_lock:
            processors[cam_id] = preloaded_processor
            processors_initializing.discard(cam_id)
            processor_errors.pop(cam_id, None)
        
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
            with processors_lock:
                processor = processors.get(cam_id)
                processors_initializing.discard(cam_id)
                processor_errors.pop(cam_id, None)
            if processor is not None:
                try:
                    processor.stop()
                    with processors_lock:
                        processors.pop(cam_id, None)
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
    processor = get_processor(cam_id, wait=False)
    if processor is None:
        with processors_lock:
            is_initializing = cam_id in processors_initializing
            error_message = processor_errors.get(cam_id)
        return {
            "status": "initializing" if is_initializing else ("error" if error_message else "idle"),
            "error": error_message,
            "metrics": None,
            "logs": [],
            "growth_history": [],
            "temporal_status": None
        }

    stats = processor.get_stats()
    stats["status"] = "ready"
    stats["error"] = None
    return stats


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)

