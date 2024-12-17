import os
from dotenv import load_dotenv

# Load environment variables from a .env file
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    # Basic Flask Config
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-12345')  # Use default if SECRET_KEY is not set
    UPLOAD_FOLDER = os.path.join(basedir, 'instance')  # Folder for uploaded files
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Limit file size to 16MB

    # Dialogflow Config (if needed in your application)
    DIALOGFLOW_PROJECT_ID = os.getenv('DIALOGFLOW_PROJECT_ID')

    # Application Data Paths
    CLUSTERING_DATA_PATH = os.path.join(basedir, 'instance', 'download-data-stunting-balita-batang-per-pebruari-2023-_1_.csv')
    HEALTH_DATA_PATH = os.path.join(basedir, 'instance', 'Banyaknya Tenaga Medis Menurut Fasilitas Kesehatan di Kabupaten Batang, 2023.csv')

    # Add other relevant configurations here if needed
