from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """
    Route untuk halaman utama.
    """
    return render_template('index.html')

@main_bp.route('/dashboard')
def dashboard():
    """
    Route untuk halaman dashboard analisis.
    """
    return render_template('analysis/dashboard.html')

@main_bp.route('/chatbot')
def chatbot():
    """
    Route untuk halaman chatbot.
    """
    return render_template('chatbot/chat.html')

@main_bp.route('/vision')
def vision():
    """
    Route untuk halaman vision.
    """
    return render_template('vision/vision.html')


@main_bp.route('/upload')
def upload():
    """
    Route untuk halaman upload.
    """
    return render_template('upload/upload.html')