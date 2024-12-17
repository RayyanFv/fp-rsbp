from flask import Blueprint, render_template, request, jsonify
from app.services.dialogflow_service import DialogflowService
import uuid

chatbot_bp = Blueprint('chatbot', __name__)
dialogflow_service = DialogflowService()

@chatbot_bp.route('/')
def chatbot():
    """
    Halaman utama chatbot.
    """
    return render_template('chatbot/chat.html')

@chatbot_bp.route('/send_message', methods=['POST'])
def send_message():
    """
    Endpoint untuk menangani pesan dari frontend.
    """
    try:
        # Ambil data JSON dari permintaan
        data = request.get_json()
        print(f"[DEBUG] Data diterima: {data}")

        if not data or 'message' not in data:
            return jsonify({'error': 'Invalid request. "message" is required.'}), 400

        message = data.get('message')
        session_id = data.get('session_id') or str(uuid.uuid4())
        print(f"[DEBUG] Message: {message}, Session ID: {session_id}")

        # Kirim pesan ke Dialogflow
        response = dialogflow_service.detect_intent(session_id, message)
        print(f"[DEBUG] Response dari Dialogflow: {response}")

        if response:
            return jsonify({
                'response': response['fulfillment_text'],
                'intent': response['intent'],
                'confidence': response['confidence'],
                'session_id': session_id
            })
        else:
            return jsonify({'error': 'Failed to get a valid response from Dialogflow'}), 500

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan di send_message: {e}")
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500
