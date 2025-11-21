# ai_kyc_full_admin_final_fixed.py
"""
AI-KYC — Full merged single-file Flask app (fixed admin action endpoint)

This is your provided final file with the admin action endpoint renamed to
`admin_action` to match templates and avoid BuildError.
"""

import os
import io
import re
import sqlite3
import hashlib
import datetime
from functools import wraps

from flask import Flask, request, render_template_string, redirect, url_for, send_file, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import pytesseract
import numpy as np
import cv2

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
DB_PATH = "kyc_admin.db"
ALLOWED_EXT = {"png", "jpg", "jpeg"}
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # change if needed

# App UI title (keeps previous Bootstrap UI look)
APP_TITLE = "AI-KYC — Admin Dashboard"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# configure pytesseract path on Windows
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except Exception:
    pass

app = Flask(__name__)
app.secret_key = "replace-with-strong-secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------- Backends detection -------------------
USE_FACE_RECOG = False
USE_DEEPFACE = False

try:
    import face_recognition
    USE_FACE_RECOG = True
    print("[INFO] face_recognition available (dlib).")
except Exception as e:
    print("[INFO] face_recognition not available:", e)
    USE_FACE_RECOG = False

try:
    from deepface import DeepFace
    USE_DEEPFACE = True
    print("[INFO] DeepFace available.")
except Exception as e:
    print("[INFO] DeepFace not available:", e)
    USE_DEEPFACE = False

# ------------------- DB Migration -------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def migrate_db():
    conn = get_conn()
    c = conn.cursor()
    # Create table with all columns expected by UI and code
    c.execute("""
    CREATE TABLE IF NOT EXISTS applicants (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT,
      dob TEXT,
      id_filename TEXT,
      selfie_filename TEXT,
      action_selfie_filename TEXT DEFAULT NULL,
      ocr_text TEXT,
      face_conf REAL,
      liveness_score REAL DEFAULT NULL,
      blur_score REAL DEFAULT NULL,
      risk REAL,
      ai_suggested TEXT,
      ai_reasons TEXT,
      final_status TEXT,
      created_at TEXT
    );
    """)
    # admins table
    c.execute("""
    CREATE TABLE IF NOT EXISTS admins (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE,
      password_hash TEXT,
      created_at TEXT
    );
    """)
    # audit logs
    c.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      admin_username TEXT,
      action TEXT,
      app_id INTEGER,
      note TEXT,
      ts TEXT
    );
    """)

    # ensure default admin exists
    c.execute("SELECT id FROM admins WHERE username=?", ("admin",))
    if not c.fetchone():
        ph = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute("INSERT INTO admins (username,password_hash,created_at) VALUES (?,?,?)",
                  ("admin", ph, datetime.datetime.utcnow().isoformat()))
        print("[INIT] Created default admin admin/admin123")

    conn.commit()
    conn.close()

migrate_db()

# ------------------- Utilities -------------------
def allowed_file(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() in ALLOWED_EXT

def extract_text(path):
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        print("OCR error:", e)
        return ""

# robust name + DOB extraction for different formats
def naive_name_dob(ocr):
    lines = [l.strip() for l in (ocr or "").splitlines() if l.strip()]
    name = ""
    dob = ""
    dob_patterns = [
        r"\d{2}[./-]\d{2}[./-]\d{4}",
        r"\d{2}[./-]\d{2}[./-]\d{2}",
        r"\d{4}[./-]\d{2}[./-]\d{2}",
        r"\d{1,2}\s+[A-Z]{3,}\s+\d{4}"
    ]
    # labeled search
    for l in lines[:40]:
        L = l.upper()
        if "NAME" in L and ":" in l:
            candidate = l.split(":",1)[1].strip()
            if candidate:
                name = candidate
        if any(k in L for k in ("DOB","DATE OF BIRTH","BIRTH")) and not dob:
            for p in dob_patterns:
                m = re.search(p, l, flags=re.IGNORECASE)
                if m:
                    dob = m.group(0)
                    break
        if name and dob:
            break
    # fallback dob anywhere
    if not dob:
        for l in lines:
            for p in dob_patterns:
                m = re.search(p, l, flags=re.IGNORECASE)
                if m:
                    dob = m.group(0); break
            if dob: break
    # fallback name: first non-digit-ish line (top few)
    if not name and lines:
        for l in lines[:6]:
            if re.search(r'\d', l): continue
            # skip address-like words
            if any(w in l.upper() for w in ("ADDRESS","GOVT","INDIA","PIN","LICENSE","LICENCE")): continue
            if len(l.split()) >= 2:
                name = l; break
    return name, dob

# ------------------- Face matching -------------------
def face_match_face_recognition(id_path, selfie_path):
    try:
        id_img = face_recognition.load_image_file(id_path)
        sf_img = face_recognition.load_image_file(selfie_path)
        enc1 = face_recognition.face_encodings(id_img)
        enc2 = face_recognition.face_encodings(sf_img)
        if not enc1 or not enc2:
            return None
        d = np.linalg.norm(enc1[0] - enc2[0])
        conf = max(0.0, 1.0 - d/0.6)  # tuned threshold mapping
        return round(float(conf), 3)
    except Exception as e:
        print("FR error:", e)
        return None

def face_match_deepface(id_path, selfie_path):
    try:
        res = DeepFace.verify(img1_path=id_path, img2_path=selfie_path, enforce_detection=False)
        dist = res.get("distance", 1.0)
        conf = max(0.0, 1.0 - dist)
        return round(float(conf),3)
    except Exception as e:
        print("DeepFace error:", e)
        return None

def face_match_histogram(id_path, selfie_path):
    try:
        f = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        i1 = cv2.imread(id_path, cv2.IMREAD_GRAYSCALE)
        i2 = cv2.imread(selfie_path, cv2.IMREAD_GRAYSCALE)
        if i1 is None or i2 is None:
            return None
        f1 = f.detectMultiScale(i1, scaleFactor=1.1, minNeighbors=4)
        f2 = f.detectMultiScale(i2, scaleFactor=1.1, minNeighbors=4)
        if len(f1)==0 or len(f2)==0:
            return None
        x,y,w,h = f1[0]
        c1 = cv2.resize(i1[y:y+h, x:x+w], (150,150))
        x,y,w,h = f2[0]
        c2 = cv2.resize(i2[y:y+h, x:x+w], (150,150))
        h1 = cv2.calcHist([c1],[0],None,[256],[0,256])
        h2 = cv2.calcHist([c2],[0],None,[256],[0,256])
        cv2.normalize(h1,h1); cv2.normalize(h2,h2)
        sim = cv2.compareHist(h1,h2,cv2.HISTCMP_CORREL)
        conf = (sim + 1)/2
        return round(float(conf),3)
    except Exception as e:
        print("histogram error:", e)
        return None

def compute_face_conf(id_path, selfie_path):
    # priority: face_recognition -> deepface -> histogram
    if USE_FACE_RECOG:
        out = face_match_face_recognition(id_path, selfie_path)
        if out is not None: return out
    if USE_DEEPFACE:
        out = face_match_deepface(id_path, selfie_path)
        if out is not None: return out
    return face_match_histogram(id_path, selfie_path)

# ------------------- Liveness & blur heuristics -------------------
def liveness_estimate(image_path):
    # Simple heuristic placeholder: check if eyes detected (Haar) and not grayscale-only
    try:
        img = cv2.imread(image_path)
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return 0.0
        x,y,w,h = faces[0]
        roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi)
        # if eyes found, higher liveness score
        score = 1.0 if len(eyes) >= 1 else 0.6
        return float(score)
    except Exception as e:
        print("liveness error:", e)
        return None

def blur_score(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        fm = cv2.Laplacian(img, cv2.CV_64F).var()
        # map variance of Laplacian to 0..1 where lower means blur
        # threshold 100 is often used; scale accordingly
        s = max(0.0, min(1.0, fm / 200.0))
        return round(1.0 - s, 3)  # return blur severity (0 good, 1 very blurry)
    except Exception as e:
        print("blur error:", e)
        return None

# ------------------- Risk & AI suggestion -------------------
def compute_risk_and_reasons(ocr_text, face_conf, liveness, blur, total_size):
    reasons = []
    score = 0.0
    if not ocr_text or len(ocr_text.strip()) < 30:
        score += 0.4
        reasons.append("OCR text too short / unclear")
    if face_conf is None:
        score += 0.6
        reasons.append("Face not detected / match failed")
    else:
        if face_conf < 0.45:
            score += 0.6
            reasons.append(f"Low face match ({face_conf})")
        elif face_conf < 0.6:
            score += 0.3
            reasons.append(f"Moderate face match ({face_conf})")
        else:
            reasons.append(f"Good face match ({face_conf})")
    if liveness is None or liveness < 0.5:
        score += 0.2
        reasons.append("Low liveness score")
    if blur is not None and blur > 0.5:
        score += 0.2
        reasons.append("Image appears blurry")
    if total_size < 5000:
        score += 0.1
        reasons.append("Images very small")
    score = round(min(1.0, score), 3)
    return score, reasons

def ai_suggest(risk, face_conf, liveness, ocr_text):
    # Conservative suggestion logic
    if face_conf is not None and face_conf >= 0.65 and risk < 0.35 and liveness is not None and liveness >= 0.8:
        return "APPROVED"
    if risk < 0.6:
        return "PENDING"
    return "FLAGGED"

# ------------------- PDF export -------------------
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def make_pdf(row):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w,h = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, h-50, f"KYC Report — Application #{row['id']}")
    c.setFont("Helvetica", 11)
    y = h-90
    items = [
        ("Name", row.get("name")),
        ("DOB", row.get("dob")),
        ("Face Confidence", str(row.get("face_conf"))),
        ("Liveness", str(row.get("liveness_score"))),
        ("Blur (severity)", str(row.get("blur_score"))),
        ("Risk", str(row.get("risk"))),
        ("AI suggested", row.get("ai_suggested")),
        ("AI reasons", row.get("ai_reasons")),
        ("Final status", row.get("final_status")),
        ("Created at", row.get("created_at"))
    ]
    for k,v in items:
        c.drawString(40, y, f"{k}: {v}")
        y -= 16
    # embed images small
    try:
        idp = os.path.join(UPLOAD_FOLDER, row.get("id_filename") or "")
        sfn = os.path.join(UPLOAD_FOLDER, row.get("selfie_filename") or "")
        if os.path.exists(idp):
            c.drawImage(idp, 40, y-160, width=200, height=120)
        if os.path.exists(sfn):
            c.drawImage(sfn, 260, y-160, width=200, height=120)
    except Exception as e:
        print("pdf embed error", e)
    c.showPage()
    c.save(); buf.seek(0)
    return buf

# ------------------- TEMPLATES (same UI style as earlier) -------------------
LOGIN_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Admin Login - {{title}}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-5">
  <div class="row justify-content-center">
    <div class="col-md-6">
      <div class="card shadow-sm">
        <div class="card-body">
          <h4 class="card-title mb-3">Admin Login</h4>
          {% with messages = get_flashed_messages() %}
            {% if messages %}
              <div class="alert alert-warning">{{ messages[0] }}</div>
            {% endif %}
          {% endwith %}
          <form method="post" action="{{ url_for('admin_login') }}">
            <div class="mb-3">
              <label class="form-label">Username</label>
              <input class="form-control" name="username" required>
            </div>
            <div class="mb-3">
              <label class="form-label">Password</label>
              <input type="password" class="form-control" name="password" required>
            </div>
            <div class="d-grid">
              <button class="btn btn-primary">Sign in</button>
            </div>
          </form>
        </div>
      </div>
      <p class="text-muted small mt-2">Default admin/admin123 (change later)</p>
    </div>
  </div>
</div>
</body>
</html>
"""

ADMIN_DASH_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{{title}}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>.nowrap{white-space:nowrap}</style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark bg-primary">
  <div class="container">
    <a class="navbar-brand" href="{{ url_for('admin_dashboard') }}">{{title}}</a>
    <div class="ms-auto">
      <span class="text-white me-3">Admin: {{session.get('admin_username')}}</span>
      <a class="btn btn-outline-light btn-sm" href="{{ url_for('admin_logout') }}">Logout</a>
    </div>
  </div>
</nav>
<div class="container my-4">
  <div class="row mb-3">
    <div class="col-md-8">
      <form class="d-flex" method="get">
        <input class="form-control me-2" name="q" placeholder="Search name or status" value="{{ request.args.get('q','') }}">
        <button class="btn btn-outline-secondary">Search</button>
      </form>
    </div>
    <div class="col-md-4 text-end">
      <a class="btn btn-success" href="{{ url_for('index') }}">Go to User Upload</a>
    </div>
  </div>

  <div class="row mb-2">
    <div class="col">
      <div class="card p-3">
        <div class="row text-center">
          <div class="col"><h5 class="mb-0">{{stats.total}}</h5><small class="text-muted">Total</small></div>
          <div class="col"><h5 class="mb-0 text-success">{{stats.approved}}</h5><small class="text-muted">Approved</small></div>
          <div class="col"><h5 class="mb-0 text-warning">{{stats.pending}}</h5><small class="text-muted">Pending</small></div>
          <div class="col"><h5 class="mb-0 text-danger">{{stats.flagged}}</h5><small class="text-muted">Flagged</small></div>
        </div>
      </div>
    </div>
  </div>

  <table class="table table-striped align-middle">
    <thead>
      <tr>
        <th>ID</th><th>Name</th><th>AI</th><th>Face</th><th>Liveness</th><th>Risk</th><th>Final</th><th>Created</th><th class="text-end">Actions</th>
      </tr>
    </thead>
    <tbody>
    {% for a in apps %}
      <tr>
        <td>{{a['id']}}</td>
        <td>{{a['name'] or 'Unknown'}}</td>
        <td class="nowrap">{{a['ai_suggested']}}</td>
        <td>{{a['face_conf']}}</td>
        <td>{{a['liveness_score']}}</td>
        <td>{{a['risk']}}</td>
        <td>
          {% if a['final_status']=='APPROVED' %}<span class="badge bg-success">APPROVED</span>
          {% elif a['final_status']=='FLAGGED' %}<span class="badge bg-danger">FLAGGED</span>
          {% else %}<span class="badge bg-secondary">{{a['final_status'] or 'PENDING'}}</span>{% endif %}
        </td>
        <td>{{a['created_at'][:19]}}</td>
        <td class="text-end nowrap">
          <a class="btn btn-sm btn-outline-primary" href="{{ url_for('view_app_admin', app_id=a['id']) }}">View</a>
          <a class="btn btn-sm btn-outline-success" href="{{ url_for('admin_action', app_id=a['id'], action='approve') }}">Approve</a>
          <a class="btn btn-sm btn-outline-warning" href="{{ url_for('admin_action', app_id=a['id'], action='pending') }}">Pending</a>
          <a class="btn btn-sm btn-outline-danger" href="{{ url_for('admin_action', app_id=a['id'], action='reject') }}">Reject</a>
        </td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

</div>
</body>
</html>
"""

USER_INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AI-KYC Upload</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-4">
  <div class="row">
    <div class="col-md-6">
      <div class="card p-3 mb-3">
        <h5>Upload ID + Selfie</h5>
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            <div class="alert alert-info">{{ messages[0] }}</div>
          {% endif %}
        {% endwith %}
        <form method="post" action="{{ url_for('submit_user') }}" enctype="multipart/form-data">
          <div class="mb-2">
            <label class="form-label">ID Document (PNG/JPG)</label>
            <input class="form-control" type="file" name="id_doc" required>
          </div>
          <div class="mb-2">
            <label class="form-label">Selfie (PNG/JPG)</label>
            <input class="form-control" type="file" name="selfie" required>
          </div>
          <div class="mb-2">
            <label class="form-label">Name (optional)</label>
            <input class="form-control" name="name">
          </div>
          <button class="btn btn-primary">Submit for KYC</button>
        </form>
      </div>

      <h5>Recent Submissions</h5>
      <div class="list-group">
        {% for a in recent %}
          <div class="list-group-item">
            <div class="d-flex w-100 justify-content-between">
              <h6 class="mb-1">ID: {{a['id']}} — {{a['name'] or 'Unknown'}}</h6>
              <small>{{a['created_at'][:19]}}</small>
            </div>
            <p class="mb-1">AI Suggestion: <strong>{{a['ai_suggested']}}</strong> — Final: <strong>{{a['final_status'] or 'PENDING'}}</strong></p>
            <a href="{{ url_for('view_app_public', app_id=a['id']) }}" class="btn btn-sm btn-outline-secondary">View</a>
          </div>
        {% endfor %}
      </div>

    </div>
    <div class="col-md-6">
      <div class="card p-3">
        <h5>About</h5>
        <p>This prototype uses OCR + face-match + heuristics. Admin finalizes decisions.</p>
        <a class="btn btn-sm btn-success" href="{{ url_for('admin_login') }}">Admin Login</a>
      </div>
    </div>
  </div>
</div>
</body>
</html>
"""

VIEW_PUBLIC = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Application #{{app['id']}}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"></head>
<body class="bg-light">
<div class="container py-4">
  <div class="card p-3">
    <h4>Application #{{app['id']}}</h4>
    <p><strong>Name:</strong> {{app['name'] or 'N/A'}}</p>
    <p><strong>DOB:</strong> {{app['dob'] or 'N/A'}}</p>
    <p><strong>OCR snippet:</strong><pre>{{(app['ocr_text'] or '')[:400]}}</pre></p>
    <p><strong>Face match:</strong> {{app['face_conf']}}</p>
    <p><strong>Liveness:</strong> {{app['liveness_score']}}</p>
    <p><strong>AI suggested:</strong> {{app['ai_suggested']}}</p>
    <p><strong>Final:</strong> {{app['final_status'] or 'PENDING'}}</p>
    <p>
      <a class="btn btn-outline-primary btn-sm" href="{{ url_for('download_file', app_id=app['id'], which='id') }}">Download ID</a>
      <a class="btn btn-outline-primary btn-sm" href="{{ url_for('download_file', app_id=app['id'], which='selfie') }}">Download Selfie</a>
      <a class="btn btn-outline-secondary btn-sm" href="{{ url_for('download_pdf', app_id=app['id']) }}">Download PDF</a>
    </p>
    <a href="{{ url_for('index') }}" class="btn btn-link">Back</a>
  </div>
</div>
</body>
</html>
"""

VIEW_ADMIN = """
<!doctype html>
<html><head><meta charset="utf-8"><title>Admin View</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"></head>
<body>
<div class="container py-4">
  <div class="card p-3">
    <h4>Application #{{app['id']}}</h4>
    <p><strong>Name:</strong> {{app['name'] or 'N/A'}}</p>
    <p><strong>DOB:</strong> {{app['dob'] or 'N/A'}}</p>
    <p><strong>OCR:</strong><pre>{{(app['ocr_text'] or '')[:800]}}</pre></p>
    <p><strong>Face Confidence:</strong> {{app['face_conf']}}</p>
    <p><strong>Liveness:</strong> {{app['liveness_score']}}</p>
    <p><strong>Blur (severity):</strong> {{app['blur_score']}}</p>
    <p><strong>AI suggestion:</strong> {{app['ai_suggested']}}</p>
    <p><strong>AI reasons:</strong> {{app['ai_reasons']}}</p>
    <p><strong>Final status:</strong> {{app['final_status'] or 'PENDING'}}</p>
    <p>
      <a class="btn btn-success" href="{{ url_for('admin_action', app_id=app['id'], action='approve') }}">Approve</a>
      <a class="btn btn-warning" href="{{ url_for('admin_action', app_id=app['id'], action='pending') }}">Pending</a>
      <a class="btn btn-danger" href="{{ url_for('admin_action', app_id=app['id'], action='reject') }}">Reject</a>
      <a class="btn btn-outline-secondary" href="{{ url_for('download_file', app_id=app['id'], which='id') }}">Download ID</a>
      <a class="btn btn-outline-secondary" href="{{ url_for('download_file', app_id=app['id'], which='selfie') }}">Download Selfie</a>
      <a class="btn btn-outline-secondary" href="{{ url_for('download_pdf', app_id=app['id']) }}">PDF</a>
      <a class="btn btn-outline-danger" href="{{ url_for('admin_delete', app_id=app['id']) }}">Delete</a>
    </p>
    <a href="{{ url_for('admin_dashboard') }}" class="btn btn-link">Back</a>
  </div>
</div>
</body>
</html>
"""

# ------------------- Routes: Public -------------------
@app.route("/")
def index():
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT id,name,dob,id_filename,selfie_filename,ocr_text,face_conf,liveness_score,blur_score,risk,ai_suggested,ai_reasons,final_status,created_at FROM applicants ORDER BY id DESC LIMIT 8")
    rows = c.fetchall(); conn.close()
    recent = [dict(r) for r in rows]
    return render_template_string(USER_INDEX_HTML, recent=recent)

@app.route("/submit", methods=["POST"])
def submit_user():
    if 'id_doc' not in request.files or 'selfie' not in request.files:
        flash("Both files are required"); return redirect(url_for('index'))
    idf = request.files['id_doc']; sf = request.files['selfie']
    if idf.filename=='' or sf.filename=='': flash("Missing"); return redirect(url_for('index'))
    if not allowed_file(idf.filename) or not allowed_file(sf.filename): flash("Use png/jpg/jpeg"); return redirect(url_for('index'))

    ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    id_fn = secure_filename(f"{ts}_id_{idf.filename}")
    sf_fn = secure_filename(f"{ts}_self_{sf.filename}")
    id_path = os.path.join(UPLOAD_FOLDER, id_fn)
    sf_path = os.path.join(UPLOAD_FOLDER, sf_fn)

    id_bytes = idf.read(); sf_bytes = sf.read()
    with open(id_path,"wb") as f: f.write(id_bytes)
    with open(sf_path,"wb") as f: f.write(sf_bytes)

    ocr_text = extract_text(id_path)
    name_auto, dob = naive_name_dob(ocr_text)
    name_field = (request.form.get("name") or "").strip()
    final_name = name_field or name_auto

    face_conf = compute_face_conf(id_path, sf_path)
    live_score = liveness_estimate(sf_path)
    blur = blur_score(sf_path)
    total_size = len(id_bytes) + len(sf_bytes)
    risk, reasons = compute_risk_and_reasons(ocr_text, face_conf, live_score, blur, total_size)
    ai_sugg = ai_suggest(risk, face_conf, live_score, ocr_text)
    created = datetime.datetime.utcnow().isoformat()

    conn = get_conn(); c = conn.cursor()
    c.execute("""INSERT INTO applicants
        (name,dob,id_filename,selfie_filename,ocr_text,face_conf,liveness_score,blur_score,risk,ai_suggested,ai_reasons,final_status,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (final_name, dob, id_fn, sf_fn, ocr_text, face_conf, live_score, blur, risk, ai_sugg, "; ".join(reasons), None, created))
    conn.commit(); app_id = c.lastrowid; conn.close()

    flash(f"Submitted. Application id: {app_id} — AI suggested: {ai_sugg}")
    return redirect(url_for('view_app_public', app_id=app_id))

@app.route("/app/<int:app_id>")
def view_app_public(app_id):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT * FROM applicants WHERE id=?", (app_id,))
    r = c.fetchone(); conn.close()
    if not r: return "Not found", 404
    return render_template_string(VIEW_PUBLIC, app=dict(r))

@app.route("/download/<int:app_id>/<which>")
def download_file(app_id, which):
    if which not in ("id","selfie"): return "Invalid",400
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT id_filename,selfie_filename FROM applicants WHERE id=?", (app_id,))
    r = c.fetchone(); conn.close()
    if not r: return "Not found",404
    fn = r[0] if which=="id" else r[1]
    path = os.path.join(UPLOAD_FOLDER, fn)
    if not os.path.exists(path): return "File missing",404
    return send_file(path, as_attachment=True)

@app.route("/pdf/<int:app_id>")
def download_pdf(app_id):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT * FROM applicants WHERE id=?", (app_id,))
    r = c.fetchone(); conn.close()
    if not r: return "Not found",404
    buf = make_pdf(dict(r))
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=f"kyc_{app_id}.pdf")

# ------------------- Admin auth & pages -------------------
@app.route("/admin/login", methods=["GET","POST"])
def admin_login():
    if request.method=="GET":
        return render_template_string(LOGIN_HTML, title=APP_TITLE)
    u = request.form.get("username","").strip()
    p = request.form.get("password","").strip()
    if not u or not p:
        flash("Provide username & password"); return redirect(url_for('admin_login'))
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT password_hash FROM admins WHERE username=?", (u,))
    r = c.fetchone(); conn.close()
    if not r:
        flash("Invalid credentials"); return redirect(url_for('admin_login'))
    ph = r['password_hash']
    if hashlib.sha256(p.encode()).hexdigest() == ph:
        session['admin_username'] = u
        return redirect(url_for('admin_dashboard'))
    else:
        flash("Invalid credentials"); return redirect(url_for('admin_login'))

@app.route("/admin/logout")
def admin_logout():
    session.pop('admin_username', None)
    return redirect(url_for('index'))

def admin_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not session.get('admin_username'):
            return redirect(url_for('admin_login'))
        return f(*a, **kw)
    return wrapper

@app.route("/admin")
@admin_required
def admin_dashboard():
    q = request.args.get('q','').strip()
    conn = get_conn(); c = conn.cursor()
    base_sql = "SELECT id,name,dob,id_filename,selfie_filename,ocr_text,face_conf,liveness_score,blur_score,risk,ai_suggested,ai_reasons,final_status,created_at FROM applicants"
    params = ()
    if q:
        like = f"%{q}%"
        base_sql += " WHERE name LIKE ? OR ai_suggested LIKE ? OR final_status LIKE ?"
        params = (like, like, like)
    base_sql += " ORDER BY id DESC"
    c.execute(base_sql, params)
    rows = c.fetchall()
    # stats
    c.execute("SELECT COUNT(*) FROM applicants"); total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM applicants WHERE final_status='APPROVED'"); approved = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM applicants WHERE final_status='PENDING' OR final_status IS NULL"); pending = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM applicants WHERE final_status='FLAGGED'"); flagged = c.fetchone()[0]
    conn.close()
    stats = {"total":total,"approved":approved,"pending":pending,"flagged":flagged}
    apps = [dict(r) for r in rows]
    return render_template_string(ADMIN_DASH_HTML, title=APP_TITLE, apps=apps, stats=stats)

@app.route("/admin/view/<int:app_id>")
@admin_required
def view_app_admin(app_id):
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT * FROM applicants WHERE id=?", (app_id,))
    r = c.fetchone(); conn.close()
    if not r: return "Not found",404
    return render_template_string(VIEW_ADMIN, app=dict(r))

# ---- FIXED: single admin_action endpoint (matches templates) ----
@app.route("/admin/action/<int:app_id>/<action>")
@admin_required
def admin_action(app_id, action):
    # action: approve / pending / reject
    if action not in ("approve","pending","reject"):
        flash("Invalid action"); return redirect(url_for('admin_dashboard'))
    final = {"approve":"APPROVED","pending":"PENDING","reject":"FLAGGED"}[action]
    admin_user = session.get('admin_username')
    conn = get_conn(); c = conn.cursor()
    c.execute("UPDATE applicants SET final_status=? WHERE id=?", (final, app_id))
    c.execute("INSERT INTO audit_logs (admin_username,action,app_id,note,ts) VALUES (?,?,?,?,?)",
              (admin_user, action, app_id, f"Set final_status={final}", datetime.datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    flash(f"Set application {app_id} -> {final}")
    return redirect(url_for('admin_dashboard'))

@app.route("/admin/delete/<int:app_id>")
@admin_required
def admin_delete(app_id):
    admin_user = session.get('admin_username')
    conn = get_conn(); c = conn.cursor()
    c.execute("SELECT id_filename,selfie_filename FROM applicants WHERE id=?", (app_id,))
    r = c.fetchone()
    if r:
        try:
            if r['id_filename']:
                os.remove(os.path.join(UPLOAD_FOLDER, r['id_filename']))
            if r['selfie_filename']:
                os.remove(os.path.join(UPLOAD_FOLDER, r['selfie_filename']))
        except Exception:
            pass
    c.execute("DELETE FROM applicants WHERE id=?", (app_id,))
    c.execute("INSERT INTO audit_logs (admin_username,action,app_id,note,ts) VALUES (?,?,?,?,?)",
              (admin_user, "delete", app_id, "Deleted application and files", datetime.datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    flash("Deleted")
    return redirect(url_for('admin_dashboard'))

# ------------------- Run -------------------
if __name__ == "__main__":
    print("[START] AI-KYC final starting...")
    print("[CONFIG] Upload folder:", UPLOAD_FOLDER, "DB:", DB_PATH)
    print("[INFO] USE_FACE_RECOG =", USE_FACE_RECOG, "USE_DEEPFACE =", USE_DEEPFACE)
    app.run(debug=True)
