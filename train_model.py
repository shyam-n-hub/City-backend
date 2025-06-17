import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# Enhanced dataset with more diverse and realistic descriptions
data = {
    "text": [
        # Pothole descriptions
        "big pothole on main road causing damage to vehicles",
        "deep hole in the road near school",
        "road surface damaged with large crater",
        "pothole filled with water creating hazard",
        "multiple potholes on highway affecting traffic",
        "road depression causing vehicle damage",
        "asphalt broken creating hole in street",
        "crater like hole in the middle of road",
        "road surface cracked and broken",
        "dangerous pothole near traffic signal",
        "road has big holes everywhere",
        "pavement damaged with deep holes",
        "street surface broken badly",
        "road full of potholes and cracks",
        "huge pothole blocking vehicle movement",
        
        # Streetlight descriptions
        "streetlight not working since last week",
        "lamp post is broken and not lighting",
        "street lamp flickering constantly",
        "no light on the street at night",
        "streetlight pole is damaged",
        "bulb not working in street lamp",
        "dark street due to faulty streetlight",
        "street lighting system is down",
        "lamp post needs repair urgently",
        "streetlight switch not functioning",
        "no illumination on road at night",
        "street lamp making strange noise",
        "light pole is tilted and dangerous",
        "streetlight cable is hanging loose",
        "area is too dark without proper lighting",
        
        # Garbage descriptions
        "garbage pile near residential area",
        "trash dumped on street corner",
        "overflowing dustbin attracting flies",
        "waste scattered all over the road",
        "garbage truck didnt collect waste",
        "pile of rubbish creating bad smell",
        "trash bags torn and waste spread",
        "garbage bin is full and overflowing",
        "waste disposal area is messy",
        "litter scattered on the street",
        "garbage accumulated near market",
        "trash not collected for many days",
        "waste bin broken and spilling garbage",
        "debris and waste blocking walkway",
        "garbage dump creating health hazard",
        
        # Water Leakage descriptions
        "water pipe burst on main road",
        "sewage leaking from manhole cover",
        "water leakage from underground pipe",
        "drainage pipe broken causing flood",
        "water supply line damaged",
        "sewage overflow on the street",
        "water main break causing flooding",
        "leakage from water distribution pipe",
        "broken water pipeline flooding area",
        "sewage backup in residential area",
        "water gushing from broken pipe",
        "drainage system overflow during rain",
        "water leak from municipal pipeline",
        "sewage pipe burst creating mess",
        "water supply disruption due to leakage",
        
        # Mixed/Complex descriptions that could be multiple categories
        "pothole filled with sewage water",
        "streetlight not working and garbage piled nearby",
        "broken road with water logging",
        "lamp post damaged due to water leakage",
        "garbage thrown in potholes",
        "street flooded due to blocked drainage",
        "waste water overflowing on damaged road",
        "dark street with potholes and garbage",
        "water leak causing road surface damage",
        "trash scattered near broken streetlight",
        
        # Additional specific cases
        "road maintenance required immediately",
        "street cleaning needed urgently",
        "public lighting system failure",
        "water infrastructure damage",
        "sanitation issue in neighborhood",
        "municipal services not working",
        "civic amenities need attention",
        "public infrastructure broken",
        "utility services disrupted",
        "community facilities damaged"
    ],
    "label": [
        # Pothole labels
        "Pothole", "Pothole", "Pothole", "Pothole", "Pothole",
        "Pothole", "Pothole", "Pothole", "Pothole", "Pothole",
        "Pothole", "Pothole", "Pothole", "Pothole", "Pothole",
        
        # Streetlight labels
        "Streetlight", "Streetlight", "Streetlight", "Streetlight", "Streetlight",
        "Streetlight", "Streetlight", "Streetlight", "Streetlight", "Streetlight",
        "Streetlight", "Streetlight", "Streetlight", "Streetlight", "Streetlight",
        
        # Garbage labels
        "Garbage", "Garbage", "Garbage", "Garbage", "Garbage",
        "Garbage", "Garbage", "Garbage", "Garbage", "Garbage",
        "Garbage", "Garbage", "Garbage", "Garbage", "Garbage",
        
        # Water Leakage labels
        "Water Leakage", "Water Leakage", "Water Leakage", "Water Leakage", "Water Leakage",
        "Water Leakage", "Water Leakage", "Water Leakage", "Water Leakage", "Water Leakage",
        "Water Leakage", "Water Leakage", "Water Leakage", "Water Leakage", "Water Leakage",
        
        # Mixed/Complex cases - assign primary category
        "Water Leakage", "Garbage", "Pothole", "Streetlight", "Garbage",
        "Water Leakage", "Water Leakage", "Garbage", "Water Leakage", "Garbage",
        
        # Additional cases
        "Pothole", "Garbage", "Streetlight", "Water Leakage", "Garbage",
        "Garbage", "Pothole", "Pothole", "Streetlight", "Pothole"
    ]
}

def train_enhanced_model():
    """Train an enhanced model with better accuracy and feature extraction"""
    
    # Create DataFrame
    df = pd.DataFrame(data)
    print(f"Training with {len(df)} samples")
    print(f"Categories: {df['label'].value_counts().to_dict()}")
    
    # Enhanced TF-IDF Vectorizer with better parameters
    vectorizer = TfidfVectorizer(
        max_features=1000,           # Increased vocabulary size
        ngram_range=(1, 3),          # Include bigrams and trigrams
        stop_words='english',        # Remove common English stop words
        lowercase=True,              # Convert to lowercase
        token_pattern=r'\b[a-zA-Z]+\b',  # Only alphabetic tokens
        min_df=1,                    # Minimum document frequency
        max_df=0.95,                 # Maximum document frequency
        sublinear_tf=True            # Use sublinear term frequency scaling
    )
    
    # Transform text data
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train enhanced Naive Bayes model
    model = MultinomialNB(alpha=0.1)  # Reduced alpha for less smoothing
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Test with some sample predictions
    test_samples = [
        "huge hole in the road",
        "street lamp not working",
        "garbage everywhere",
        "water pipe leaking",
        "broken streetlight and garbage nearby"
    ]
    
    print(f"\nSample Predictions:")
    for sample in test_samples:
        X_sample = vectorizer.transform([sample])
        prediction = model.predict(X_sample)[0]
        confidence = max(model.predict_proba(X_sample)[0])
        print(f"'{sample}' -> {prediction} (confidence: {confidence:.3f})")
    
    # Save model and vectorizer
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    
    print(f"\nModel and vectorizer saved successfully!")
    print(f"Files created: model.pkl, vectorizer.pkl")
    
    # Feature importance analysis
    print(f"\nTop features for each category:")
    feature_names = vectorizer.get_feature_names_out()
    
    for i, category in enumerate(model.classes_):
        # Get top features for this category
        top_features_idx = model.feature_log_prob_[i].argsort()[-10:][::-1]
        top_features = [feature_names[idx] for idx in top_features_idx]
        print(f"{category}: {', '.join(top_features)}")
    
    return model, vectorizer

def validate_model():
    """Additional validation with edge cases"""
    try:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        
        # Test edge cases
        edge_cases = [
            "",  # Empty string
            "help",  # Very short
            "there is a problem in my area",  # Vague description  
            "road pothole streetlight garbage water",  # Multiple keywords
            "urgent repair needed immediately",  # Generic urgent request
        ]
        
        print(f"\nEdge Case Testing:")
        for case in edge_cases:
            try:
                if case.strip():
                    X_case = vectorizer.transform([case])
                    prediction = model.predict(X_case)[0]
                    confidence = max(model.predict_proba(X_case)[0])
                    print(f"'{case}' -> {prediction} (confidence: {confidence:.3f})")
                else:
                    print(f"'{case}' -> Empty input handled")
            except Exception as e:
                print(f"'{case}' -> Error: {e}")
                
    except FileNotFoundError:
        print("Model files not found. Please run train_enhanced_model() first.")

if __name__ == "__main__":
    print("Starting Enhanced Model Training...")
    model, vectorizer = train_enhanced_model()
    
    print("\nRunning additional validation...")
    validate_model()
    
    print("\nTraining completed successfully!")
    print("The model is now ready for use in your Flask backend.")