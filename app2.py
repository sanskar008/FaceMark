import os
import face_recognition
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

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


# Create the database (run once)
with app.app_context():
    db.create_all()


# Function: Load registered student encodings from the database
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

    # Save the uploaded image
    image_path = os.path.join(STUDENTS_DIR, f"{name}.jpg")
    if os.path.exists(image_path):
        return jsonify({'error': 'A student with this name already exists'}), 400

    image.save(image_path)

    # Get face encoding for the uploaded image
    loaded_image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(loaded_image)
    if encodings:
        encoding = encodings[0]
        try:
            student = Student(name=name, encoding=encoding)
            db.session.add(student)
            db.session.commit()
            return jsonify({'message': f'Student {name} registered successfully'}), 200
        except Exception as e:
            return jsonify({'error': 'Failed to register student. Name may already exist.'}), 400
    else:
        os.remove(image_path)  # Remove the image file if no face is detected
        return jsonify({'error': 'No face detected in the image'}), 400


# Endpoint: Mark attendance for multiple faces
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

    # Compare with registered students and handle multiple faces
    results = set()
    for uploaded_encoding in uploaded_encodings:
        matches = face_recognition.compare_faces(known_encodings, uploaded_encoding, tolerance=0.6)
        distances = face_recognition.face_distance(known_encodings, uploaded_encoding)

        if True in matches:
            match_index = np.argmin(distances)
            results.add(student_names[match_index])

    if results:
        return jsonify({'message': f'Attendance marked for: {", ".join(results)}'}), 200
    else:
        return jsonify({'error': 'No match found for the given faces'}), 404


# Endpoint: List all registered students
@app.route('/students', methods=['GET'])
def get_students():
    students = Student.query.all()
    result = [{'id': student.id, 'name': student.name} for student in students]
    return jsonify(result), 200


# Endpoint: Homepage
@app.route('/')
def home():
    return render_template('index1.html')


if __name__ == '__main__':
    app.run(debug=True)
