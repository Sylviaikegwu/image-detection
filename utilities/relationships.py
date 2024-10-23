def compute_relationships(detections):
    """Compute relationships between detected objects."""
    relationships = []
    for i in range(len(detections)):
        for j in range(len(detections)):
            if i != j:
                relationship = {
                    'subject': detections[i]['class'],
                    'object': detections[j]['class'],
                    'relationship': 'interacts_with'  
                }
                relationships.append(relationship)
    return relationships

def analyze_scene_features(detections):
    """Analyze global scene features based on detections."""
    scene_features = {
        'total_objects': len(detections),
        'object_classes': [det['class'] for det in detections],
        'common_relationships': []
    }
    
    for det in detections:
        if det['class'] not in scene_features['common_relationships']:
            scene_features['common_relationships'].append(det['class'])
    
    return scene_features
