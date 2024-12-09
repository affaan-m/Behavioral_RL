import os

def create_directory_structure():
    """Create the project directory structure"""
    directories = [
        'data/web_results',
        'data/model_checkpoints',
        'data/analysis',
        'logs/training',
        'logs/experiments',
        'src/templates',
        'src/static/css',
        'src/static/js',
        'src/visualization',
        'metrics'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_directory_structure()
    print("\nDirectory structure setup complete!") 