from google.cloud import dialogflow
import os

class DialogflowService:
    def __init__(self):
        try:
            # Path ke file JSON kredensial
            credentials_path = "instance/retinoassistant-hhhk-0151ed5d6f62.json"
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Credential file not found: {credentials_path}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

            # ID proyek dari file JSON
            self.project_id = "retinoassistant-hhhk"
            self.session_client = dialogflow.SessionsClient()
            self.language_code = "id"  # Gunakan Bahasa Indonesia
            print("[INFO] DialogflowService initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize DialogflowService: {e}")
            raise RuntimeError("Failed to initialize DialogflowService") from e

    def detect_intent(self, session_id, text):
        """
        Mengirimkan teks pengguna ke Dialogflow untuk mendeteksi intent.
        """
        try:
            session = self.session_client.session_path(self.project_id, session_id)
            text_input = dialogflow.TextInput(text=text, language_code=self.language_code)
            query_input = dialogflow.QueryInput(text=text_input)

            response = self.session_client.detect_intent(
                request={"session": session, "query_input": query_input}
            )
            print(f"[DEBUG] Response dari Dialogflow: {response}")
            return {
                'fulfillment_text': response.query_result.fulfillment_text,
                'intent': response.query_result.intent.display_name,
                'confidence': response.query_result.intent_detection_confidence
            }
        except Exception as e:
            print(f"[ERROR] Failed to detect intent: {e}")
            return None
