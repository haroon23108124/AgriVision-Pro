import os
import joblib 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from features import extract_features 


DATASET_DIR = r"C:\Users\haroo\Desktop\PlantVillage" 
MODEL_NAME = "plant_disease_rf_model.pkl"

def main():
    print("--- STARTING TRAINING ---")
    
    X = [] 
    y = [] #class names
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset folder '{DATASET_DIR}' not found.")
        return

    classes = os.listdir(DATASET_DIR)
    print(f"Found classes: {classes}")

    print("Extracting features from images ...")
    for class_name in classes:
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"Processing class: {class_name}...")
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
           
            img_features = extract_features(img_path)
            
            if img_features is not None:
                X.append(img_features)
                y.append(class_name)

    X = np.array(X)
    y = np.array(y)
    
    print(f"\nData extraction complete. Total images: {len(X)}")
    
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Step 2: Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    
    print("Step 3: Evaluating Model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- RESULTS ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))
    
    
    joblib.dump(clf, MODEL_NAME)
    print(f"\nModel saved successfully as '{MODEL_NAME}'")

if __name__ == "__main__":
    main()