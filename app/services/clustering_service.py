import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import folium
import os
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

class ClusteringService:
    """
    Service untuk melakukan analisis clustering pada data stunting.
    
    Attributes:
        app: Flask application instance
        stunting_df: DataFrame untuk data stunting mentah
        kecamatan_data: DataFrame untuk data yang sudah diagregasi per kecamatan
        clustering_results: Dictionary berisi hasil clustering
        kecamatan_coordinates: Dictionary koordinat setiap kecamatan
        logger: Logger instance untuk tracking error dan info
    """
    
    def __init__(self, app=None):
        self.app = app
        self.stunting_df = None
        self.kecamatan_data = None
        self.clustering_results = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Koordinat kecamatan (divalidasi)
        self.kecamatan_coordinates = {
            'BATANG': {'lat': -6.9075, 'lon': 109.7268},
            'WONOTUNGGAL': {'lat': -7.0180, 'lon': 109.8012},
            'BANDAR': {'lat': -7.0382, 'lon': 109.8307},
            'BLADO': {'lat': -7.0598, 'lon': 109.8383},
            'REBAN': {'lat': -7.0962, 'lon': 109.8876},
            'BAWANG': {'lat': -7.0991, 'lon': 109.9121},
            'TERSONO': {'lat': -7.0640, 'lon': 109.8815},
            'GRINGSING': {'lat': -6.9493, 'lon': 109.9597},
            'LIMPUNG': {'lat': -6.9995, 'lon': 109.9001},
            'BANYUPUTIH': {'lat': -6.9743, 'lon': 109.9668},
            'SUBAH': {'lat': -6.9717, 'lon': 109.8815},
            'PECALUNGAN': {'lat': -7.0134, 'lon': 109.8534},
            'TULIS': {'lat': -6.9307, 'lon': 109.8524},
            'KANDEMAN': {'lat': -6.9329, 'lon': 109.7673},
            'WARUNGASEM': {'lat': -6.9343, 'lon': 109.7890}
        }

    def load_data(self) -> bool:
        """
        Load dan validasi data stunting dari CSV.
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            file_path = os.path.join(
                self.app.instance_path,
                'download-data-stunting-balita-batang-per-pebruari-2023-_1_.csv'
            )
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return False
                
            self.stunting_df = pd.read_csv(file_path)
            
            # Validasi kolom yang dibutuhkan
            required_columns = ['NAMA_KECAMATAN', 'JUMLAH_BALITA', 
                              'JUMLAH_BALITA_SANGAT_PENDEK', 'JUMLAH_BALITA_PENDEK']
            if not all(col in self.stunting_df.columns for col in required_columns):
                self.logger.error("Missing required columns in CSV file")
                return False
                
            return self.preprocess_data()
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False

    def preprocess_data(self) -> bool:
        """
        Preprocessing data stunting dengan handling missing values dan outliers.
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            # Convert dan validasi kolom numerik
            numeric_cols = ['JUMLAH_BALITA', 'JUMLAH_BALITA_SANGAT_PENDEK', 'JUMLAH_BALITA_PENDEK']
            for col in numeric_cols:
                # Convert to numeric and handle missing values
                self.stunting_df[col] = pd.to_numeric(self.stunting_df[col], errors='coerce')
                
                # Check for negative values
                if (self.stunting_df[col] < 0).any():
                    self.logger.warning(f"Negative values found in {col}")
                    self.stunting_df[col] = self.stunting_df[col].clip(lower=0)
                
                # Fill missing values with median
                median_value = self.stunting_df[col].median()
                self.stunting_df[col] = self.stunting_df[col].fillna(median_value)

            # Calculate stunting metrics
            self.stunting_df['total_stunting'] = (
                self.stunting_df['JUMLAH_BALITA_SANGAT_PENDEK'] + 
                self.stunting_df['JUMLAH_BALITA_PENDEK']
            )
            
            # Validate total_stunting tidak melebihi JUMLAH_BALITA
            invalid_rows = self.stunting_df['total_stunting'] > self.stunting_df['JUMLAH_BALITA']
            if invalid_rows.any():
                self.logger.warning("Found rows where total stunting exceeds total children")
                self.stunting_df.loc[invalid_rows, 'total_stunting'] = self.stunting_df.loc[invalid_rows, 'JUMLAH_BALITA']

            # Calculate percentages
            self.stunting_df['stunting_percentage'] = (
                self.stunting_df['total_stunting'] / 
                self.stunting_df['JUMLAH_BALITA'] * 100
            ).round(2)

            # Aggregate by kecamatan dengan validasi
            self.kecamatan_data = self.stunting_df.groupby('NAMA_KECAMATAN').agg({
                'JUMLAH_BALITA': 'sum',
                'total_stunting': 'sum'
            }).reset_index()

            # Validate kecamatan names
            invalid_kecamatan = set(self.kecamatan_data['NAMA_KECAMATAN']) - set(self.kecamatan_coordinates.keys())
            if invalid_kecamatan:
                self.logger.warning(f"Unknown kecamatan found: {invalid_kecamatan}")

            self.kecamatan_data['stunting_percentage'] = (
                self.kecamatan_data['total_stunting'] /
                self.kecamatan_data['JUMLAH_BALITA'] * 100
            ).round(2)

            # Handle outliers using IQR method
            Q1 = self.kecamatan_data['stunting_percentage'].quantile(0.25)
            Q3 = self.kecamatan_data['stunting_percentage'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.kecamatan_data[
                (self.kecamatan_data['stunting_percentage'] < lower_bound) |
                (self.kecamatan_data['stunting_percentage'] > upper_bound)
            ]
            
            if not outliers.empty:
                self.logger.warning(f"Found outliers in stunting percentage: {outliers['NAMA_KECAMATAN'].tolist()}")

            return True
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            return False

    def find_optimal_clusters(self, max_k: int = 5) -> int:
        """
        Menentukan jumlah cluster optimal menggunakan multiple metrics.
        
        Args:
            max_k: Maximum number of clusters to test
            
        Returns:
            int: Optimal number of clusters
        """
        try:
            features = ['stunting_percentage', 'JUMLAH_BALITA']
            X = self.kecamatan_data[features].copy()
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate metrics
            metrics = {
                'distortions': [],      # Elbow method
                'silhouette': [],       # Silhouette score
                'calinski': []          # Calinski-Harabasz score
            }
            
            K = range(2, max_k + 1)
            
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                
                metrics['distortions'].append(kmeans.inertia_)
                metrics['silhouette'].append(silhouette_score(X_scaled, kmeans.labels_))
                metrics['calinski'].append(calinski_harabasz_score(X_scaled, kmeans.labels_))

            # Plot metrics
            plt.figure(figsize=(15, 5))
            
            # Elbow Method
            plt.subplot(1, 3, 1)
            plt.plot(K, metrics['distortions'], 'bx-')
            plt.xlabel('k')
            plt.ylabel('Distortion')
            plt.title('Elbow Method')

            # Silhouette Score
            plt.subplot(1, 3, 2)
            plt.plot(K, metrics['silhouette'], 'rx-')
            plt.xlabel('k')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score Method')

            # Calinski-Harabasz Score
            plt.subplot(1, 3, 3)
            plt.plot(K, metrics['calinski'], 'gx-')
            plt.xlabel('k')
            plt.ylabel('Calinski-Harabasz Score')
            plt.title('Calinski-Harabasz Score')

            # Save plot
            plot_path = os.path.join(self.app.static_folder, 'img', 'clustering_metrics.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            # Determine optimal k using voting
            optimal_k_silhouette = K[np.argmax(metrics['silhouette'])]
            optimal_k_calinski = K[np.argmax(metrics['calinski'])]
            
            # Use elbow method: find point of maximum curvature
            distortions_array = np.array(metrics['distortions'])
            diffs = np.diff(distortions_array)
            optimal_k_elbow = K[np.argmax(np.abs(np.diff(diffs))) + 1]
            
            # Voting system
            votes = [optimal_k_silhouette, optimal_k_calinski, optimal_k_elbow]
            optimal_k = max(set(votes), key=votes.count)  # Most common value
            
            self.logger.info(f"Optimal k suggestions - Silhouette: {optimal_k_silhouette}, "
                           f"Calinski-Harabasz: {optimal_k_calinski}, Elbow: {optimal_k_elbow}")
            self.logger.info(f"Selected optimal k: {optimal_k}")
            
            return optimal_k

        except Exception as e:
            self.logger.error(f"Error finding optimal clusters: {str(e)}")
            return 3  # Default fallback

    def perform_clustering(self, n_clusters: Optional[int] = None) -> Optional[List[int]]:
        """
        Melakukan analisis clustering dengan validasi dan evaluasi.
        
        Args:
            n_clusters: Number of clusters (optional, will find optimal if None)
            
        Returns:
            List[int]: Cluster labels atau None jika gagal
        """
        try:
            if n_clusters is None:
                n_clusters = self.find_optimal_clusters()
                
            features = ['stunting_percentage', 'JUMLAH_BALITA']
            X = self.kecamatan_data[features].copy()
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform clustering with multiple initializations
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # Evaluate clustering quality
            silhouette_avg = silhouette_score(X_scaled, labels)
            calinski_score = calinski_harabasz_score(X_scaled, labels)
            
            self.logger.info(f"Clustering metrics - Silhouette: {silhouette_avg:.3f}, "
                           f"Calinski-Harabasz: {calinski_score:.3f}")

            # Store results
            self.clustering_results = {
                'labels': [int(label) for label in labels],
                'centroids': kmeans.cluster_centers_.tolist(),
                'inertia': float(kmeans.inertia_),
                'silhouette_score': float(silhouette_avg),
                'calinski_score': float(calinski_score)
            }

            # Add cluster labels to kecamatan_data
            self.kecamatan_data['cluster'] = self.clustering_results['labels']

            # Visualize results
            plt.figure(figsize=(12, 8))
            
            # Main scatter plot
            scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                c=labels, cmap='viridis', s=100)
            
            # Plot centroids
            plt.scatter(kmeans.cluster_centers_[:, 0], 
                       kmeans.cluster_centers_[:, 1], 
                       marker='x', s=200, linewidths=3,
                       color='r', label='Centroids')
            
            # Add kecamatan labels
            for idx, row in self.kecamatan_data.iterrows():
                plt.annotate(row['NAMA_KECAMATAN'], 
                           (X_scaled[idx, 0], X_scaled[idx, 1]),
                           xytext=(5, 5), textcoords='offset points')
            
            plt.title('Hasil Clustering Stunting per Kecamatan')
            plt.xlabel('Standarisasi Persentase Stunting')
            plt.ylabel('Standarisasi Jumlah Balita')
            plt.colorbar(scatter, label='Cluster')
            plt.legend()

            # Save plot
            plot_path = os.path.join(self.app.static_folder, 'img', 'clustering_results.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            return self.clustering_results['labels']
            
        except Exception as e:
            self.logger.error(f"Error performing clustering: {str(e)}")
            return None

    def create_map(self) -> bool:
        """
        Membuat peta interaktif dengan hasil clustering.
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            # Initialize map centered on Batang
            m = folium.Map(
                location=[-6.9075, 109.7268],
                zoom_start=11,
                min_zoom=10,
                max_zoom=13,
                tiles='CartoDB positron'  # Clean, modern basemap
            )

            # Define color scheme and risk levels
            colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']  # ColorBrewer RdYlGn
            risk_levels = ['Sangat Tinggi', 'Tinggi', 'Sedang', 'Rendah', 'Sangat Rendah']

            # Calculate cluster statistics for risk assessment
            cluster_stats = {}
            for cluster in range(len(set(self.clustering_results['labels']))):
                mask = self.kecamatan_data['cluster'] == cluster
                cluster_stats[cluster] = {
                    'avg_stunting': self.kecamatan_data.loc[mask, 'stunting_percentage'].mean(),
                    'total_balita': self.kecamatan_data.loc[mask, 'JUMLAH_BALITA'].sum(),
                    'kecamatan_count': mask.sum()
                }

            # Sort clusters by risk (average stunting percentage)
            sorted_clusters = sorted(cluster_stats.items(), 
                                  key=lambda x: x[1]['avg_stunting'], 
                                  reverse=True)
            
            # Map clusters to risk levels and colors
            cluster_risk_map = {cluster: risk for (cluster, _), risk in 
                              zip(sorted_clusters, risk_levels[:len(sorted_clusters)])}
            cluster_color_map = {cluster: color for (cluster, _), color in 
                               zip(sorted_clusters, colors[:len(sorted_clusters)])}

            # Add markers for each kecamatan with enhanced tooltips
            for idx, row in self.kecamatan_data.iterrows():
                kecamatan = row['NAMA_KECAMATAN']
                if kecamatan in self.kecamatan_coordinates:
                    coord = self.kecamatan_coordinates[kecamatan]
                    cluster = row['cluster']
                    color = cluster_color_map[cluster]
                    risk_level = cluster_risk_map[cluster]

                    # Enhanced popup with statistics and styling
                    popup_content = f"""
                    <div style='min-width: 200px; font-family: Arial, sans-serif;'>
                        <h4 style='margin-bottom: 10px; color: #2c3e50;'>{kecamatan}</h4>
                        <div style='background-color: {color}; color: white; padding: 5px; 
                                  border-radius: 3px; margin-bottom: 10px;'>
                            <b>Tingkat Risiko:</b> {risk_level}
                        </div>
                        <div style='line-height: 1.5;'>
                            <b>Persentase Stunting:</b> {row['stunting_percentage']:.1f}%<br>
                            <b>Jumlah Balita:</b> {int(row['JUMLAH_BALITA']):,}<br>
                            <b>Total Stunting:</b> {int(row['total_stunting']):,}<br>
                        </div>
                    </div>
                    """

                    # Create circle marker with popup
                    folium.CircleMarker(
                        location=[coord['lat'], coord['lon']],
                        radius=10 + (row['stunting_percentage'] / 5),  # Size based on percentage
                        popup=folium.Popup(popup_content, max_width=300),
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        weight=2,
                        tooltip=f"{kecamatan} - {risk_level}"
                    ).add_to(m)

            # Add enhanced legend with statistics
            legend_html = """
            <div style="position: fixed; bottom: 50px; left: 50px; z-index:1000; 
                        background-color: white; padding: 15px; border-radius: 5px;
                        box-shadow: 0 0 15px rgba(0,0,0,0.2); max-width: 300px;">
                <h4 style='margin-bottom: 10px; border-bottom: 2px solid #2c3e50; 
                          padding-bottom: 5px;'>Tingkat Risiko Stunting</h4>
            """
            
            for cluster, stats in sorted_clusters:
                color = cluster_color_map[cluster]
                risk = cluster_risk_map[cluster]
                legend_html += f"""
                <div style='margin-bottom: 5px;'>
                    <span style='color: {color}; font-size: 20px;'>●</span>
                    <b>{risk}</b><br>
                    <div style='margin-left: 20px; font-size: 0.9em;'>
                        Rata-rata: {stats['avg_stunting']:.1f}%<br>
                        Jumlah Kecamatan: {stats['kecamatan_count']}
                    </div>
                </div>
                """
            legend_html += "</div>"
            m.get_root().html.add_child(folium.Element(legend_html))

            # Save map
            save_path = os.path.join(self.app.static_folder, 'map.html')
            m.save(save_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating map: {str(e)}")
            return False

    def get_cluster_statistics(self) -> Dict:
        """
        Mendapatkan statistik detail untuk setiap cluster.
        
        Returns:
            Dict: Statistik per cluster
        """
        try:
            stats = {}
            for cluster in range(len(set(self.clustering_results['labels']))):
                cluster_data = self.kecamatan_data[self.kecamatan_data['cluster'] == cluster]
                
                # Calculate additional statistics
                stunting_percentages = cluster_data['stunting_percentage']
                balita_counts = cluster_data['JUMLAH_BALITA']
                
                stats[f'Cluster {cluster}'] = {
                    'Jumlah Kecamatan': int(len(cluster_data)),
                    'Rata-rata Stunting': f"{stunting_percentages.mean():.1f}%",
                    'Median Stunting': f"{stunting_percentages.median():.1f}%",
                    'Std Dev Stunting': f"{stunting_percentages.std():.1f}%",
                    'Min Stunting': f"{stunting_percentages.min():.1f}%",
                    'Max Stunting': f"{stunting_percentages.max():.1f}%",
                    'Total Balita': int(balita_counts.sum()),
                    'Rata-rata Balita': int(balita_counts.mean()),
                    'Total Stunting': int(cluster_data['total_stunting'].sum()),
                    'Kecamatan': sorted(cluster_data['NAMA_KECAMATAN'].tolist())
                }
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting cluster statistics: {str(e)}")
            return {}

    def get_summary(self) -> Dict:
        """
        Mendapatkan ringkasan statistik keseluruhan.
        
        Returns:
            Dict: Ringkasan statistik
        """
        try:
            summary = {
                'total_kecamatan': int(len(self.kecamatan_data)),
                'total_balita': int(self.kecamatan_data['JUMLAH_BALITA'].sum()),
                'total_stunting': int(self.kecamatan_data['total_stunting'].sum()),
                'avg_stunting': f"{self.kecamatan_data['stunting_percentage'].mean():.1f}%",
                'median_stunting': f"{self.kecamatan_data['stunting_percentage'].median():.1f}%",
                'std_dev_stunting': f"{self.kecamatan_data['stunting_percentage'].std():.1f}%",
                'max_stunting': f"{self.kecamatan_data['stunting_percentage'].max():.1f}%",
                'min_stunting': f"{self.kecamatan_data['stunting_percentage'].min():.1f}%",
                'cluster_count': len(set(self.clustering_results['labels'])) if self.clustering_results else 0,
                'clustering_quality': {
                    'silhouette_score': f"{self.clustering_results['silhouette_score']:.3f}",
                    'calinski_score': f"{self.clustering_results['calinski_score']:.3f}"
                } if self.clustering_results else None
            }
            
            # Add high-risk areas
            high_risk_threshold = self.kecamatan_data['stunting_percentage'].mean() + \
                                self.kecamatan_data['stunting_percentage'].std()
            high_risk_areas = self.kecamatan_data[
                self.kecamatan_data['stunting_percentage'] > high_risk_threshold
            ]['NAMA_KECAMATAN'].tolist()
            
            summary['high_risk_areas'] = sorted(high_risk_areas)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting summary: {str(e)}")
            return {}