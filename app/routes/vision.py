from flask import Blueprint, render_template, request, jsonify
from app.services.vision_service import VisionService

# Create blueprint
vision_bp = Blueprint('vision', __name__)

# Initialize vision service
vision_service = VisionService()

@vision_bp.route('/')
def index():
    """Render the main page"""
    return render_template('vision.html')

@vision_bp.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests"""
    try:
        # Get JSON data
        data = request.get_json()
        print("Received request data type:", type(data))
        
        if not data or 'image' not in data:
            print("Missing image data in request")
            return jsonify({
                'error': 'No image data provided'
            }), 400
            
        print("Image data length:", len(data['image']) if isinstance(data.get('image'), str) else "not a string")
            
        # Get prediction from service
        result = vision_service.predict(data['image'])
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500