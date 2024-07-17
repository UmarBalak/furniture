from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the YOLO model
model = YOLO('yolov9s.pt')

furniture_classes_list = [
    "bench", "chair", "couch", "potted plant", 
    "bed", "dining table", "clock", "vase"
]

complementary_items = {
    # (same as your provided complementary_items dictionary)
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            detected_items = detect_furniture(filepath)
            top_items = get_top_complementary_items(detected_items)
            return render_template('results.html', detected_items=detected_items, top_items=top_items)
    return render_template('index.html')

def detect_furniture(image_path):
    # Run predictions
    results = model.predict(image_path)
    
    # Filter detected classes
    detected_classes = []
    for result in results:
        for cls in result.boxes.cls:
            class_name = model.names[int(cls)]
            if class_name in furniture_classes_list:
                detected_classes.append(class_name)

    return detected_classes

def get_top_complementary_items(detected_items):
    item_counts = {}
    for item in detected_items:
        if item in complementary_items:
            for comp_item, count in complementary_items[item]:
                if comp_item in item_counts:
                    item_counts[comp_item] += count
                else:
                    item_counts[comp_item] = count
    
    # Remove detected items from suggestions
    for item in detected_items:
        if item in item_counts:
            del item_counts[item]

    # Sort items by count and return the top 5
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items[:5]]

if __name__ == '__main__':
    app.run(debug=True)
