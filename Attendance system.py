

#Nouman ahmad


#...............................................................................................................
# lib import
import cv2
import numpy as np
import json
import sqlite3
import datetime
from flask import Flask, render_template_string, Response, request, redirect, url_for, session
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt






#...............................................................................................................

# training phase....

def train_model():
    dataset_path = 'attendance_dataset'

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax', kernel_regularizer=l2(0.001))
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

    # Save the model and class mapping
    model.save("face_recognition_model.h5")
    class_indices = train_generator.class_indices
    inv_map = {v: k for k, v in class_indices.items()}
    with open("class_mapping.json", "w") as f:
        json.dump(inv_map, f)

    print("Training complete and model saved.")

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

    return model, inv_map
#database setup and maerking attendence.....
def init_db():

    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    # sql
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT,
        date TEXT,
        time TEXT
    )''')
    conn.commit()
    conn.close()

#inserting attendence record...
def mark_attendance(student_id):

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT * FROM attendance WHERE student_id=? AND date=?", (student_id, date_str))
    result = c.fetchone()
    if result is None:
        c.execute("INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)",
                  (student_id, date_str, time_str))
        conn.commit()
        print(f"Attendance marked for {student_id} at {date_str} {time_str}")
    conn.close()


# Flask web app
app = Flask(__name__)
app.secret_key = 'admin_key'  # 

# Load the trained model and class mapping 
model = load_model("face_recognition_model.h5")
with open("class_mapping.json", "r") as f:
    inv_map = json.load(f)

attendance_set = set()

# Face recognition and attendance marking gen videoooo
def gen_frames():

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    while True:
        success, frame = cap.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (224, 224))
            face_roi = face_roi.astype("float32") / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)

            preds = model.predict(face_roi)
            class_id = np.argmax(preds)
            label = inv_map[str(class_id)]

            # Mark attendance if not already marked in this session
            if label not in attendance_set:
                mark_attendance(label)
                attendance_set.add(label)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


#===================================================================================================================
#===================================================================================================================
# flask routing....
attendance_html = """
<!doctype html>
<html>
<head>
    <title>Student Attendance</title>
</head>
<body>
    <h1>Student Attendance</h1>
    <img src="{{ url_for('video_feed') }}" width="640" height="480">
    <p>The system will automatically mark attendance when your face is recognized.</p>
    <a href="{{ url_for('admin') }}">Admin Login</a>
</body>
</html>
"""

admin_login_html = """
<!doctype html>
<html>
<head>
    <title>Admin Login</title>
</head>
<body>
    <h1>Admin Login</h1>
    <form method="post" action="{{ url_for('admin') }}">
        <input type="password" name="password" placeholder="Enter admin password" required>
        <input type="submit" value="Login">
    </form>
</body>
</html>
"""




#===================================================================================================================
admin_dashboard_html = """
<!doctype html>
<html>
<head>
    <title>Admin Dashboard</title>
</head>
<body>
    <h1>Attendance Records</h1>
    <table border="1">
        <tr>
            <th>ID</th>
            <th>Student ID</th>
            <th>Date</th>
            <th>Time</th>
        </tr>
        {% for record in records %}
        <tr>
            <td>{{ record[0] }}</td>
            <td>{{ record[1] }}</td>
            <td>{{ record[2] }}</td>
            <td>{{ record[3] }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="{{ url_for('logout') }}">Logout</a>
</body>
</html>
"""

#===================================================================================================================
#===================================================================================================================
@app.route('/')
def index():
    return redirect(url_for('attendance'))


@app.route('/attendance')
def attendance():
    return render_template_string(attendance_html)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#===================================================================================================================
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == "admin123":  # Simple password 
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return "Invalid password", 401
    return render_template_string(admin_login_html)



#===================================================================================================================

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('admin'))
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT * FROM attendance")
    records = c.fetchall()
    conn.close()
    return render_template_string(admin_dashboard_html, records=records)



#===================================================================================================================
@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect(url_for('index'))

#===================================================================================================================

# main function...
if __name__ == "__main__":
    # intilized db....
    init_db()

    train_model()

    # flaks app run...
    app.run(host='0.0.0.0', port=5000, debug=True)
