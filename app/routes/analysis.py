# app/routes/analysis.py
from flask import Blueprint, render_template, jsonify, current_app

from app.services.clustering_service import ClusteringService
import os

analysis_bp = Blueprint('analysis', __name__, url_prefix='/analysis')
clustering_service = None

@analysis_bp.before_app_first_request
def initialize_service():
    global clustering_service
    clustering_service = ClusteringService(current_app)

@analysis_bp.route('/')
# @login_required
def index():
    return render_template('analysis/dashboard.html')

@analysis_bp.route('/process')
# @login_required
def process_data():
    try:
        print("Starting data processing...")  # Debug log
        
        # Load and preprocess data
        if not clustering_service.load_data():
            print("Failed to load data")  # Debug log
            return jsonify({'error': 'Failed to load data'}), 500

        # Find optimal number of clusters and perform clustering
        n_clusters = clustering_service.find_optimal_clusters()
        print(f"Found optimal clusters: {n_clusters}")  # Debug log
        
        labels = clustering_service.perform_clustering(n_clusters)
        print(f"Clustering completed with labels: {labels[:5]}...")  # Debug log
        
        if labels is None:
            print("Clustering failed")  # Debug log
            return jsonify({'error': 'Clustering failed'}), 500

        # Create map
        print("Creating map...")  # Debug log
        map_success = clustering_service.create_map()
        print(f"Map creation {'successful' if map_success else 'failed'}")  # Debug log

        if not map_success:
            return jsonify({'error': 'Failed to create map'}), 500

        # Get statistics
        stats = clustering_service.get_cluster_statistics()
        summary = clustering_service.get_summary()

        print("Processing completed successfully")  # Debug log
        return jsonify({
            'success': True,
            'optimal_k': n_clusters,
            'stats': stats,
            'summary': summary
        })

    except Exception as e:
        print(f"Process error: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/map')
# @login_required
def get_map():
    try:
        map_path = os.path.join(current_app.static_folder, 'map.html')
        print(f"Looking for map at: {map_path}")  # Debug log
        
        if not os.path.exists(map_path):
            print(f"Map file not found at {map_path}")  # Debug log
            return "Map not generated yet", 404
            
        print("Reading map file...")  # Debug log
        with open(map_path, 'r', encoding='utf-8') as f:
            map_content = f.read()
        print(f"Map content length: {len(map_content)}")  # Debug log
        return map_content
        
    except Exception as e:
        print(f"Map error: {str(e)}")  # Debug log
        return f"Error loading map: {str(e)}", 500