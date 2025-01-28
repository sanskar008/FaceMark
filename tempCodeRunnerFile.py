import os
import face_recognition
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Path to store registered student images
STUDENTS_DIR = "students"
os.makedirs(STUDENTS_DIR, exist_ok=True)

# Load registered student encodings
def load_student_encodings():
    encodings = []
    student_names = []
    for file in os.listdir(STUDENTS_DIR):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(STUDENTS_DIR, file)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Check if any faces are found
                encoding = encodings[0]
                encodings.append(encoding)
                student_names.append(file.split(".")[0])
            else:
                print(f"No faces detected in {file}, skipping...")
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

    return jsonify({'message': f'Student {name} registered successfully'})

# Endpoint: Mark attendance
@app.route('/attendance', methods=['POST'])
def mark_attendance():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400
    
    # Load student encodings
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
    uploaded_encoding = uploaded_encodings[0]
    results = face_recognition.compare_faces(known_encodings, uploaded_encoding)
    distances = face_recognition.face_distance(known_encodings, uploaded_encoding)

    # Find the closest match
    if True in results:
        match_index = np.argmin(distances)
        student_name = student_names[match_index]
        return jsonify({'message': f'Attendance marked for {student_name}'}), 200
    else:
        return jsonify({'error': 'No match found for the given face'}), 404

# Endpoint: Homepage
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
