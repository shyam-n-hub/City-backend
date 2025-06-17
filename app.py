from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

app = Flask(__name__)

# Fix CORS configuration - Allow multiple origins
CORS(app, origins=['https://city-fix.netlify.app'], 
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

def create_category_encoding(categories):
    """Create numerical encoding for categories"""
    unique_categories = list(set(categories))
    category_map = {cat: idx for idx, cat in enumerate(unique_categories)}
    return [category_map[cat] for cat in categories], category_map

def cluster_by_category_and_location(data):
    """
    Enhanced clustering that considers both category and location
    """
    if len(data) < 2:
        return [{"cluster": 0, "reports": data, "category": data[0].get('category', 'Unknown') if data else 'Unknown'}]
    
    # Group by category first
    category_groups = {}
    for report in data:
        category = report.get('category', 'Unknown')
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(report)
    
    all_clusters = []
    cluster_id = 0
    
    for category, reports in category_groups.items():
        if len(reports) == 1:
            # Single report becomes its own cluster
            all_clusters.append({
                "cluster": cluster_id,
                "category": category,
                "reports": reports,
                "lat": reports[0]['lat'],
                "lng": reports[0]['lng'],
                "priority": calculate_priority(reports)
            })
            cluster_id += 1
        else:
            # Cluster by location within the same category
            coords = np.array([[r['lat'], r['lng']] for r in reports])
            
            # Use smaller eps for tighter location clustering within same category
            db = DBSCAN(eps=0.001, min_samples=1).fit(coords)
            labels = db.labels_
            
            # Group reports by cluster labels
            clusters_in_category = {}
            for i, label in enumerate(labels):
                if label not in clusters_in_category:
                    clusters_in_category[label] = []
                clusters_in_category[label].append(reports[i])
            
            # Create cluster objects
            for label, cluster_reports in clusters_in_category.items():
                avg_lat = np.mean([r['lat'] for r in cluster_reports])
                avg_lng = np.mean([r['lng'] for r in cluster_reports])
                
                all_clusters.append({
                    "cluster": cluster_id,
                    "category": category,
                    "reports": cluster_reports,
                    "lat": float(avg_lat),
                    "lng": float(avg_lng),
                    "priority": calculate_priority(cluster_reports)
                })
                cluster_id += 1
    
    # Sort clusters by priority (high to low)
    all_clusters.sort(key=lambda x: x['priority'], reverse=True)
    
    return all_clusters

def calculate_priority(reports):
    """
    Calculate priority score for a cluster based on:
    - Number of reports
    - Status (pending gets higher priority)
    - Category importance
    """
    priority_score = 0
    
    # Base score from number of reports
    priority_score += len(reports) * 10
    
    # Status-based scoring
    for report in reports:
        status = report.get('status', 'pending')
        if status == 'pending':
            priority_score += 15
        elif status == 'doing':
            priority_score += 5
        # completed reports don't add to priority
    
    # Category-based scoring
    category_weights = {
        'Water Leakage': 20,
        'Pothole': 15,
        'Streetlight': 10,
        'Garbage': 8
    }
    
    for report in reports:
        category = report.get('category', 'Unknown')
        priority_score += category_weights.get(category, 5)
    
    return priority_score

@app.route('/cluster', methods=['POST'])
def cluster_reports():
    try:
        data = request.json
        if not data or len(data) == 0:
            return jsonify([])
        
        # Enhanced clustering
        clustered = cluster_by_category_and_location(data)
        
        # Add cluster statistics and urgency info
        for cluster in clustered:
            reports = cluster['reports']
            
            # Add cluster statistics
            cluster['stats'] = {
                'total_reports': len(reports),
                'pending': len([r for r in reports if r.get('status') == 'pending']),
                'doing': len([r for r in reports if r.get('status') == 'doing']),
                'completed': len([r for r in reports if r.get('status') == 'completed']),
                'has_images': len([r for r in reports if r.get('imageUrl')]),
                'avg_description_length': np.mean([len(r.get('description', '')) for r in reports])
            }
            
            # Add urgency level
            if cluster['priority'] >= 50:
                cluster['urgency'] = 'High'
            elif cluster['priority'] >= 25:
                cluster['urgency'] = 'Medium'
            else:
                cluster['urgency'] = 'Low'
        
        return jsonify(clustered)
    
    except Exception as e:
        print(f"Clustering error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Endpoint for dashboard analytics"""
    try:
        # This would typically fetch from your database
        # For now, returning mock data structure
        return jsonify({
            "total_reports": 0,
            "pending_reports": 0,
            "clusters_count": 0,
            "high_priority_clusters": 0,
            "category_distribution": {},
            "status_distribution": {}
        })
    except Exception as e:
        print(f"Analytics error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "Backend is working!", "timestamp": str(pd.Timestamp.now())})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)