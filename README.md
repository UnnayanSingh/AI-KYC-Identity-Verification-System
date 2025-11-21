# 🧠 AI-KYC Identity Verification System  
An AI-powered KYC (Know Your Customer) verification system that automates identity authentication using OCR, Face Recognition, Liveness Detection, Blur Analysis, and AI-driven Risk Scoring.  
Built with **Python, Flask, OpenCV, Tesseract, Deep Learning**, and a secure admin dashboard.

---

## 📌 Features

### 🔍 **Identity Processing**
- OCR-based text extraction from ID documents  
- Automatic Name & DOB extraction using regex + NLP  
- Face Recognition using DeepFace / face_recognition  
- Liveness Detection using Haar Cascades  
- Blur Detection using Laplacian variance  
- Size-based quality checks  

### 🤖 **AI Decision Engine**
- Computes a combined risk score  
- Suggests: **APPROVED / PENDING / FLAGGED**  
- Multi-factor scoring (OCR + Face + Liveness + Blur)

### 🛠️ **Admin Dashboard**
- Approve / Reject / Mark Pending  
- PDF Report Generator  
- Audit Logs  
- Search + Filter  
- Statistics (Total, Pending, Approved, Flagged)

### 🧱 **Backend**
- Python Flask Application  
- SQLite Database  
- Secure Admin Auth  
- Upload Handling + Validation

---

## 🏗️ System Architecture

```
User Upload → OCR Layer → Face Recognition → Liveness → Blur Detection → Risk Engine → AI Decision → Admin Dashboard → SQLite DB
```

### 📌 Architecture Diagram  
Add this image after adding it to your repo:
```
assets/architecture_diagram.png
```

---

## 🗂️ Folder Structure

```
AI-KYC-Identity-Verification-System/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│
├── assets/
│     └── architecture_diagram.png
│
├── docs/
│     ├── AI_KYC_Hackathon_Submission.docx
│     └── AI_KYC_Project_Report.pdf
│
├── uploads/
│     └── .gitkeep
│
├── static/
└── templates/
```

---

## 🧰 Technology Stack
- **Python 3.10+**
- **Flask**
- **OpenCV**
- **Tesseract OCR**
- **face_recognition / DeepFace**
- **NumPy, Pillow**
- **SQLite**
- **Bootstrap 5**
- **ReportLab PDF Generator**

---

## 🔧 Installation

Clone the repository:
```bash
git clone https://github.com/UnnayanSingh/AI-KYC-Identity-Verification-System.git
cd AI-KYC-Identity-Verification-System
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Windows Tesseract Setup:
```
Set TESSERACT_CMD=C:\Program Files\Tesseract-OCR	esseract.exe
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open browser at:
```
http://127.0.0.1:5000/
```

---

## 🗄️ Database Structure

### **Applicants Table**
- id, name, dob, id_image, selfie  
- face_conf, liveness, blur  
- risk_score, ai_suggestion, final_status  
- created_at  

### **Admins Table**
- id, username, password_hash, created_at  

### **Audit Logs Table**
- id, admin_username, action, app_id, timestamp  

---

## 🎥 Demo Video
(Add your YouTube / Google Drive link here)

---

## 📄 Documentation  
- [Full Project Report PDF](docs/AI_KYC_Project_Report.pdf)

---

## 👤 Author   
B.Tech – Computer Science and Engineering (Cybersecurity)

---

## 📜 License  
This project is open-source under the **MIT License**.
