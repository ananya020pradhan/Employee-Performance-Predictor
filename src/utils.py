import os

def ensure_directories():
    folders = [
        "data/raw",
        "data/processed",
        "models",
        "outputs",
        "images"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("Directories created successfully")