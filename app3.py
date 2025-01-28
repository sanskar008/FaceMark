import os
import face_recognition
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

# Set up the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Path to store registered student images
STUDENTS_DIR = "students"
os.makedirs(STUDENTS_DIR, exist_ok=True)

# Database model for storing students
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    encoding = db.Column(db.PickleType, nullable=False)

    def __repr__(self):
        return f'<Student {self.name}>'

# Database model for attendance records
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    student = db.relationship('Student', backref=db.backref('attendance_records', lazy=True))

    def __repr__(self):
        return f'<Attendance {self.student.name} at {self.timestamp}>'

# Create the database (run once)
with app.app_context():
    db.create_all()

# Load registered student encodings from the database
def load_student_encodings():
    students = Student.query.all()
    encodings = [student.encoding for student in students]
    student_names = [student.name for student in students]
    return encodings, student_names

# Endpoint: Register a new student
@app.route('/register', methods=['POST'])
def register_student():
    if 'name' not in request.form or 'image' not in request.files:
        return jsonify({'error': 'Name and image are required'}), 400
    
    name = request.form['name']
    image = request.files['image']
    image_path = os.path.join(STUDENTS_DIR, f"{name}.jpg")
    image.save(image_path)

    # Get face encoding for the uploaded image
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        encoding = encodings[0]
        student = Student(name=name, encoding=encoding)
        db.session.add(student)
        db.session.commit()
        return jsonify({'message': f'Student {name} registered successfully'}), 200
    else:
        return jsonify({'error': 'No face detected in the image'}), 400

# Endpoint: Mark attendance
@app.route('/attendance', methods=['POST'])
def mark_attendance():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400

    # Load student encodings from the database
    known_encodings, student_names = load_student_encodings()

    if not known_encodings:
        return jsonify({'error': 'No registered students found'}), 400

    # Process uploaded image
    image = request.files['image']
    uploaded_image = face_recognition.load_image_file(image)
    uploaded_encodings = face_recognition.face_encodings(uploaded_image)

    if not uploaded_encodings:
        return jsonify({'error': 'No face detected in the image'}), 400

    # Compare with registered students
    results = []
    for uploaded_encoding in uploaded_encodings:
        result = face_recognition.compare_faces(known_encodings, uploaded_encoding)
        distances = face_recognition.face_distance(known_encodings, uploaded_encoding)
        if True in result:
            match_index = np.argmin(distances)
            student_name = student_names[match_index]
            results.append(student_name)

            # Log attendance in the database
            student = Student.query.filter_by(name=student_name).first()
            if student:
                attendance_record = Attendance(student_id=student.id, timestamp=datetime.now())
                db.session.add(attendance_record)
                db.session.commit()

    if results:
        return jsonify({'message': f'Attendance marked for {", ".join(results)}'}), 200
    else:
        return jsonify({'error': 'No match found for the given face'}), 404

# Endpoint: View attendance log
@app.route('/attendance-log', methods=['GET'])
def attendance_log():
    attendance_data = (
        db.session.query(Student.name, db.func.count(Attendance.id), db.func.group_concat(Attendance.timestamp))
        .join(Attendance, Student.id == Attendance.student_id)
        .group_by(Student.id)
        .all()
    )

    response = []
    for name, count, timestamps in attendance_data:
        response.append({
            'name': name,
            'attendance_count': count,
            'timestamps': timestamps.split(','),
        })

    return render_template('attendance_log.html', records=response)

# Endpoint: Homepage
@app.route('/')
def home():
    return render_template('index4.html')

if __name__ == '__main__':
    app.run(debug=True)


#---------Commit---------
#This version of our application recognises multiple faces, stores there data in SQLite database, records the attandance logs, and has good ui