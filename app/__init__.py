from flask import Flask
from config import Config
import os

def create_app():
    # Membuat instance Flask
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Import blueprints
    from app.routes.main import main_bp
    from app.routes.analysis import analysis_bp
    from app.routes.chatbot import chatbot_bp
    from app.routes.vision import vision_bp  # Import vision blueprint
    from app.routes.vision_upload import upload_bp
    
    # Register blueprints dengan prefix URL
    app.register_blueprint(main_bp, url_prefix='/')  # Untuk rute utama
    app.register_blueprint(analysis_bp, url_prefix='/analysis')  # Untuk rute analisis
    app.register_blueprint(chatbot_bp, url_prefix='/chatbot')  # Untuk rute chatbot
    app.register_blueprint(vision_bp, url_prefix='/vision')  # Untuk rute vision
    app.register_blueprint(upload_bp, url_prefix='/upload')  # Untuk rute upload
    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
        
    return app
