"""
Script Runner - Combined batch operations for the Fake Review Detection system
"""

import os
import sys
import subprocess
import time

def print_header(title):
    """Print a formatted header"""
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)

def find_datasets_folder():
    """Find the datasets folder from current location"""
    # Check if we're in src directory
    if os.path.basename(os.getcwd()) == 'src':
        # Look for datasets in parent directory
        parent_datasets = os.path.join('..', 'datasets')
        if os.path.exists(parent_datasets):
            return parent_datasets
    
    # Check current directory
    if os.path.exists('datasets'):
        return 'datasets'
    
    # Check parent directory
    parent_datasets = os.path.join('..', 'datasets')
    if os.path.exists(parent_datasets):
        return parent_datasets
    
    return None

def choose_dataset_file():
    """Let user choose a dataset file"""
    print("Available CSV files:")
    
    # Find datasets folder
    datasets_dir = find_datasets_folder()
    csv_files = []
    file_paths = []
    
    # Priority: datasets folder first
    if datasets_dir and os.path.exists(datasets_dir):
        print(f"\n📁 Files in {datasets_dir}/ folder:")
        try:
            dataset_files = [f for f in os.listdir(datasets_dir) if f.lower().endswith('.csv')]
            for i, file in enumerate(dataset_files, 1):
                full_path = os.path.join(datasets_dir, file)
                csv_files.append(file)
                file_paths.append(full_path)
                print(f"{i}. {file}")
        except Exception as e:
            print(f"Error reading datasets folder: {e}")
    
    # Only check current directory if we're not already in datasets folder
    current_dir = os.getcwd()
    if not datasets_dir or not current_dir.endswith('datasets'):
        try:
            current_files = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
            if current_files:
                print(f"\n📁 Files in current directory:")
                start_num = len(csv_files) + 1
                for i, file in enumerate(current_files, start_num):
                    # Avoid duplicates
                    if file not in csv_files:
                        csv_files.append(file)
                        file_paths.append(file)
                        print(f"{i}. {file}")
        except Exception as e:
            print(f"Error reading current directory: {e}")
    
    if not csv_files:
        print("\n❌ No CSV files found!")
        print("Please add your labeled dataset CSV file to the 'datasets/' folder.")
        print("Expected format: text,label (where label is OR for real, CG for fake)")
        print("\nExample:")
        print("datasets/")
        print("├── labeled_fake_reviews.csv")
        print("├── amazon_reviews.csv")
        print("└── test_dataset.csv")
        return None
    
    # Add custom path option
    print(f"{len(csv_files) + 1}. Enter custom file path")
    
    # Get user choice
    while True:
        try:
            choice = input(f"\nChoose a file (1-{len(csv_files) + 1}): ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(csv_files):
                    selected_file = file_paths[choice_num - 1]
                    print(f"Selected: {selected_file}")
                    return selected_file
                elif choice_num == len(csv_files) + 1:
                    # Custom file path
                    custom_path = input("Enter the full path to your CSV file: ").strip()
                    if os.path.exists(custom_path):
                        print(f"Selected: {custom_path}")
                        return custom_path
                    else:
                        print(f"File not found: {custom_path}")
                        continue
                else:
                    print("Invalid choice. Please try again.")
                    continue
            else:
                print("Please enter a number.")
                continue
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            continue
    


def install_dependencies():
    """Install required dependencies"""
    print_header("INSTALLING DEPENDENCIES")
    
    # Check if requirements.txt exists
    requirements_path = os.path.join("..", "requirements.txt")
    if os.path.exists(requirements_path):
        print("Installing from requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    else:
        # Fallback to individual installations
        print("requirements.txt not found, installing individual packages...")
        
        # Core dependencies
        print("Installing core dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "transformers", "torch", "pandas", "scikit-learn"])
        
        # Fine-tuning dependencies
        print("Installing fine-tuning dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "accelerate"])
        
        # Visualization dependencies
        print("Installing visualization dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
    
    print("Dependencies installed successfully!")

def start_backend():
    """Start the FastAPI backend server"""
    print_header("STARTING BACKEND SERVER")
    
    print("Starting FastAPI backend on http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Change to src directory and run from there
        original_dir = os.getcwd()
        os.chdir("src")
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])
        os.chdir(original_dir)
    except KeyboardInterrupt:
        print("\nBackend server stopped.")
        os.chdir(original_dir)

def start_frontend():
    """Start the frontend (opens in browser)"""
    print_header("STARTING FRONTEND")
    
    import webbrowser
    import threading
    
    def open_browser():
        time.sleep(2)  # Wait for backend to start
        webbrowser.open("http://127.0.0.1:8000")
    
    # Start browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("Frontend will open in your browser automatically.")
    print("If it doesn't open, manually navigate to: http://127.0.0.1:8000")

def run_fine_tuning():
    """Run the fine-tuning process"""
    print_header("RUNNING FINE-TUNING")
    
    # Let user choose the dataset file
    csv_file = choose_dataset_file()
    if not csv_file:
        return False
    
    # Import and run fine-tuning
    try:
        from ml_engine import fine_tune_model, test_fine_tuned_model, get_dataset_info, get_user_range_selection
        
        # Get dataset information
        dataset_info = get_dataset_info(csv_file)
        if not dataset_info:
            return False
        
        # Get range selection from user
        start_range, end_range = get_user_range_selection(dataset_info['total_reviews'])
        if start_range is None:
            return False
        
        print(f"\nStarting fine-tuning process with: {csv_file}")
        if start_range != 0 or end_range != dataset_info['total_reviews']:
            print(f"Using range: {start_range:,} to {end_range:,} ({end_range - start_range:,} records)")
        
        results = fine_tune_model(csv_file, start_range=start_range, end_range=end_range)
        
        # Test the fine-tuned model
        test_texts = [
            "This product is absolutely amazing! Best purchase ever!",
            "The quality is terrible and I hate it completely.",
            "This is a computer generated fake review for testing purposes.",
            "I bought this item and it works as expected. Good value for money."
        ]
        
        print("\nTesting fine-tuned model...")
        test_results = test_fine_tuned_model(test_texts)
        for result in test_results:
            print(f"Text: {result['text'][:50]}...")
            print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
            print()
        
        print("Fine-tuning completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return False

def run_validation():
    """Run the validation process"""
    print_header("RUNNING VALIDATION")
    
    # Let user choose the dataset file
    csv_file = choose_dataset_file()
    if not csv_file:
        return False
    
    # Import and run validation
    try:
        from ml_engine import run_validation_pipeline, get_dataset_info, get_user_range_selection
        
        # Get dataset information
        dataset_info = get_dataset_info(csv_file)
        if not dataset_info:
            return False
        
        # Get range selection from user
        start_range, end_range = get_user_range_selection(dataset_info['total_reviews'])
        if start_range is None:
            return False
        
        print(f"\nStarting validation process with: {csv_file}")
        if start_range != 0 or end_range != dataset_info['total_reviews']:
            print(f"Using range: {start_range:,} to {end_range:,} ({end_range - start_range:,} records)")
        
        metrics = run_validation_pipeline(csv_file, start_range=start_range, end_range=end_range)
        
        if metrics:
            print("\nValidation completed successfully!")
            return True
        else:
            print("Validation failed!")
            return False
            
    except Exception as e:
        print(f"Error during validation: {e}")
        return False

def show_menu():
    """Show the main menu"""
    print_header("FAKE REVIEW DETECTION - SCRIPT RUNNER")
    print("Choose an option:")
    print("1. Install Dependencies")
    print("2. Start Backend Server")
    print("3. Start Frontend (opens in browser)")
    print("4. Run Fine-tuning")
    print("5. Run Validation")
    print("6. Start Complete System (Backend + Frontend)")
    print("7. Run Full Pipeline (Fine-tune + Validate)")
    print("0. Exit")
    print()

def run_complete_system():
    """Start the complete system (backend + frontend)"""
    print_header("STARTING COMPLETE SYSTEM")
    
    import threading
    
    def start_backend_thread():
        # Change to src directory and run from there
        original_dir = os.getcwd()
        os.chdir("src")
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])
        os.chdir(original_dir)
    
    # Start backend in background
    backend_thread = threading.Thread(target=start_backend_thread)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start frontend
    start_frontend()
    
    print("\nSystem is running!")
    print("Backend: http://127.0.0.1:8000")
    print("Frontend: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down system...")

def run_full_pipeline():
    """Run the complete pipeline (fine-tune + validate)"""
    print_header("RUNNING FULL PIPELINE")
    
    # Let user choose the dataset file once for both operations
    csv_file = choose_dataset_file()
    if not csv_file:
        return
    
    print("Step 1: Fine-tuning...")
    if run_fine_tuning_with_file(csv_file):
        print("\nStep 2: Validation...")
        if run_validation_with_file(csv_file):
            print("\nFull pipeline completed successfully!")
            print("Your fine-tuned model is ready to use!")
        else:
            print("Validation failed!")
    else:
        print("Fine-tuning failed!")

def run_fine_tuning_with_file(csv_file):
    """Run fine-tuning with a specific file"""
    try:
        from ml_engine import fine_tune_model, test_fine_tuned_model, get_dataset_info, get_user_range_selection
        
        # Get dataset information
        dataset_info = get_dataset_info(csv_file)
        if not dataset_info:
            return False
        
        # Get range selection from user
        start_range, end_range = get_user_range_selection(dataset_info['total_reviews'])
        if start_range is None:
            return False
        
        print(f"Starting fine-tuning process with: {csv_file}")
        if start_range != 0 or end_range != dataset_info['total_reviews']:
            print(f"Using range: {start_range:,} to {end_range:,} ({end_range - start_range:,} records)")
        
        results = fine_tune_model(csv_file, start_range=start_range, end_range=end_range)
        
        # Test the fine-tuned model
        test_texts = [
            "This product is absolutely amazing! Best purchase ever!",
            "The quality is terrible and I hate it completely.",
            "This is a computer generated fake review for testing purposes.",
            "I bought this item and it works as expected. Good value for money."
        ]
        
        print("\nTesting fine-tuned model...")
        test_results = test_fine_tuned_model(test_texts)
        for result in test_results:
            print(f"Text: {result['text'][:50]}...")
            print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
            print()
        
        print("Fine-tuning completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return False

def run_validation_with_file(csv_file):
    """Run validation with a specific file"""
    try:
        from ml_engine import run_validation_pipeline, get_dataset_info, get_user_range_selection
        
        # Get dataset information
        dataset_info = get_dataset_info(csv_file)
        if not dataset_info:
            return False
        
        # Get range selection from user
        start_range, end_range = get_user_range_selection(dataset_info['total_reviews'])
        if start_range is None:
            return False
        
        print(f"Starting validation process with: {csv_file}")
        if start_range != 0 or end_range != dataset_info['total_reviews']:
            print(f"Using range: {start_range:,} to {end_range:,} ({end_range - start_range:,} records)")
        
        metrics = run_validation_pipeline(csv_file, start_range=start_range, end_range=end_range)
        
        if metrics:
            print("\nValidation completed successfully!")
            return True
        else:
            print("Validation failed!")
            return False
            
    except Exception as e:
        print(f"Error during validation: {e}")
        return False

def main():
    """Main function"""
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (0-7): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                install_dependencies()
            elif choice == "2":
                start_backend()
            elif choice == "3":
                start_frontend()
            elif choice == "4":
                run_fine_tuning()
            elif choice == "5":
                run_validation()
            elif choice == "6":
                run_complete_system()
            elif choice == "7":
                run_full_pipeline()
            else:
                print("Invalid choice. Please try again.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main() 