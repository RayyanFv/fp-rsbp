from flask import Blueprint, request, render_template
from app.services.upload_service import UploadService

upload_bp = Blueprint('upload', __name__, template_folder='../templates/upload')

upload_service = UploadService()

@upload_bp.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('upload.html', error='No file part')

        file = request.files['image']

        if file.filename == '':
            return render_template('upload.html', error='No selected file')

        result = upload_service.predict(file)
        if result["status"] == "Error":
            return render_template('upload.html', error=result["message"])

        return render_template('upload.html', prediction=result["status"], confidence=result["confidence"])

    return render_template('upload.html')
