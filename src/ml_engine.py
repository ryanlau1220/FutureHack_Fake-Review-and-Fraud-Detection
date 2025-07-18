"""
Fake Review Detection - ML Engine
Combines fine-tuning, validation, and model management functionality
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
from datasets import Dataset
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

PROGRESS_FINE_TUNE_FILE = os.path.join(os.path.dirname(__file__), 'progress_fine_tune.json')
PROGRESS_VALIDATE_FILE = os.path.join(os.path.dirname(__file__), 'progress_validate.json')

def update_progress(progress_file, percent, message):
    try:
        with open(progress_file, 'w') as f:
            json.dump({'percent': percent, 'message': message}, f)
    except Exception as e:
        print(f'[Progress Error] Could not write to {progress_file}: {e}')

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def load_fake_review_model():
    """Load the appropriate fake review detection model"""
    if os.path.exists("./fine_tuned_model"):
        print("Loading fine-tuned model...")
        return pipeline("text-classification", model="./fine_tuned_model")
    else:
        print("Loading pre-trained model...")
        return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

# ============================================================================
# DATA PREPARATION
# ============================================================================

def detect_column_mapping(df):
    """Detect and map columns based on dataset format"""
    columns = df.columns.tolist()
    print(f"Detected columns: {columns}")
    
    # Initialize mapping
    text_col = None
    label_col = None
    
    # Format 1: category,rating,label,text_ (or similar)
    if 'label' in columns and any(col.lower() in ['text', 'text_', 'review', 'reviewtext'] for col in columns):
        label_col = 'label'
        for col in columns:
            if col.lower() in ['text', 'text_', 'review', 'reviewtext']:
                text_col = col
                break
    
    # Format 2: Amazon format with Text column
    elif 'Text' in columns and 'Score' in columns:
        text_col = 'Text'
        # Convert Score to binary label (1-2 = fake, 4-5 = real)
        df['label'] = df['Score'].apply(lambda x: 'CG' if x <= 2 else 'OR')
        label_col = 'label'
    
    # Format 3: Simple text,label format
    elif 'text' in columns and 'label' in columns:
        text_col = 'text'
        label_col = 'label'
    
    # Format 4: Try to find any text-like column
    else:
        for col in columns:
            if any(keyword in col.lower() for keyword in ['text', 'review', 'comment', 'content']):
                text_col = col
                break
        
        for col in columns:
            if any(keyword in col.lower() for keyword in ['label', 'class', 'category', 'type']):
                label_col = col
                break
    
    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text and label columns. Available columns: {columns}")
    
    print(f"Mapped columns: text='{text_col}', label='{label_col}'")
    return text_col, label_col

def load_and_prepare_data(csv_file_path, start_range=None, end_range=None):
    """Load and prepare the labeled dataset for fine-tuning"""
    print("Loading dataset...")
    df = pd.read_csv(csv_file_path)
    
    # Apply range selection if specified
    if start_range is not None and end_range is not None:
        print(f"Applying range selection: {start_range} to {end_range}")
        df = df.iloc[start_range:end_range]
        print(f"Selected {len(df)} reviews from range {start_range}-{end_range}")
    
    # Detect column mapping
    text_col, label_col = detect_column_mapping(df)
    
    # Clean and prepare text data first
    df['text'] = df[text_col].fillna('').astype(str)
    
    # Remove very short or very long reviews
    df = df[df['text'].str.len() > 10]
    df = df[df['text'].str.len() < 1000]
    
    # Map labels: OR (Original/Real) -> 0, CG (Computer-generated) -> 1
    label_mapping = {'OR': 0, 'OR ': 0, 'CG': 1, 'REAL': 0, 'FAKE': 1, '0': 0, '1': 1}
    
    # Handle numeric labels (Amazon format)
    if df[label_col].dtype in ['int64', 'float64']:
        # Convert numeric scores to binary (1-2 = fake, 4-5 = real)
        df['label'] = df[label_col].apply(lambda x: 1 if pd.notna(x) and x <= 2 else 0)
    else:
        # Convert string labels with better error handling
        df['label'] = df[label_col].str.upper().map(label_mapping)
        
        # Handle any unmapped values
        unmapped_mask = df['label'].isna()
        if unmapped_mask.any():
            print(f"Warning: Found {unmapped_mask.sum()} unmapped labels: {df[label_col][unmapped_mask].unique()}")
            # Remove rows with unmapped labels
            df = df[~unmapped_mask]
    
    # Final check for NaN values
    if df['label'].isna().any():
        print("Error: Still found NaN values in labels after cleaning")
        print("Label values found:", df[label_col].unique())
        raise ValueError("Dataset contains invalid labels that cannot be mapped")
    
    # Ensure labels are integers and add debugging
    df['label'] = df['label'].astype(int)
    print(f"Label mapping debug:")
    print(f"  Original labels: {df[label_col].unique()}")
    print(f"  Mapped to integers: {df['label'].unique()}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    
    print(f"Dataset loaded: {len(df)} reviews")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"Unique label values in original data: {df[label_col].unique()}")
    print(f"Label data type: {df['label'].dtype}")
    
    return df

def prepare_dataset_for_training(df, tokenizer):
    """Prepare dataset for training"""
    texts = df['text'].tolist()
    labels = df['label'].astype(int).tolist()  # Ensure labels are integers
    
    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    
    return dataset

# ============================================================================
# FINE-TUNING
# ============================================================================

def compute_metrics(pred):
    """Compute evaluation metrics for training"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def fine_tune_model(csv_file_path, model_name="mrm8488/bert-tiny-finetuned-fake-news-detection", start_range=None, end_range=None):
    """Fine-tune the fake review detection model"""
    update_progress(PROGRESS_FINE_TUNE_FILE, 0, 'Loading and preparing data...')
    df = load_and_prepare_data(csv_file_path, start_range, end_range)
    if len(df) < 10:
        update_progress(PROGRESS_FINE_TUNE_FILE, 0, f'Very small dataset ({len(df)} reviews).')
        print(f"Warning: Very small dataset ({len(df)} reviews). Fine-tuning may not be effective.")
    label_counts = df['label'].value_counts()
    if len(label_counts) < 2:
        update_progress(PROGRESS_FINE_TUNE_FILE, 0, 'Dataset only contains one class.')
        print(f"Error: Dataset only contains one class: {label_counts.index[0]}")
        print("Fine-tuning requires both fake and real reviews.")
        return None
    try:
        train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    except ValueError as e:
        update_progress(PROGRESS_FINE_TUNE_FILE, 0, 'Error during train/test split.')
        print(f"Error during train/test split: {e}")
        print("This usually happens when there are too few samples of one class.")
        print("Using random split instead...")
        train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    update_progress(PROGRESS_FINE_TUNE_FILE, 5, 'Data loaded. Preparing model...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    train_dataset = prepare_dataset_for_training(train_df, tokenizer)
    eval_dataset = prepare_dataset_for_training(eval_df, tokenizer)
    if len(train_df) < 50:
        num_epochs = 5
        batch_size = 4
        eval_steps = max(1, len(train_df) // 4)
        save_steps = max(1, len(train_df) // 2)
    else:
        num_epochs = 3
        batch_size = 8
        eval_steps = 50
        save_steps = 100
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=min(100, len(train_df) // 4),
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=max(1, len(train_df) // 10),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    update_progress(PROGRESS_FINE_TUNE_FILE, 10, 'Starting fine-tuning...')
    # Custom callback for progress
    class ProgressCallback:
        def __init__(self, total_steps):
            self.total_steps = total_steps
        def on_step_end(self, step, logs=None):
            percent = 10 + int(80 * step / self.total_steps)
            update_progress(PROGRESS_FINE_TUNE_FILE, percent, f'Training... (step {step}/{self.total_steps})')
    total_steps = int(len(train_dataset) / batch_size * num_epochs)
    progress_callback = ProgressCallback(total_steps)
    trainer.add_callback(progress_callback)
    trainer.train()
    update_progress(PROGRESS_FINE_TUNE_FILE, 90, 'Evaluating fine-tuned model...')
    eval_results = trainer.evaluate()
    update_progress(PROGRESS_FINE_TUNE_FILE, 95, 'Saving fine-tuned model...')
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    update_progress(PROGRESS_FINE_TUNE_FILE, 100, 'Fine-tuning completed!')
    print("Fine-tuning completed! Model saved to ./fine_tuned_model/")
    return eval_results

# ============================================================================
# VALIDATION AND EVALUATION
# ============================================================================

def load_data_for_validation(csv_file_path, start_range=None, end_range=None):
    """Load and prepare the dataset for validation"""
    df = pd.read_csv(csv_file_path)
    
    # Apply range selection if specified
    if start_range is not None and end_range is not None:
        print(f"Applying range selection: {start_range} to {end_range}")
        df = df.iloc[start_range:end_range]
        print(f"Selected {len(df)} reviews from range {start_range}-{end_range}")
    
    # Detect column mapping
    text_col, label_col = detect_column_mapping(df)
    
    # Clean and prepare text data first
    df['text'] = df[text_col].fillna('').astype(str)
    df = df[df['text'].str.len() > 10]
    df = df[df['text'].str.len() < 1000]
    
    # Map labels for validation with better error handling
    label_mapping = {'OR': 'REAL', 'OR ': 'REAL', 'CG': 'FAKE', 'REAL': 'REAL', 'FAKE': 'FAKE', '0': 'REAL', '1': 'FAKE'}
    
    # Handle numeric labels (Amazon format)
    if df[label_col].dtype in ['int64', 'float64']:
        # Convert numeric scores to binary (1-2 = fake, 4-5 = real)
        df['actual_label'] = df[label_col].apply(lambda x: 'FAKE' if pd.notna(x) and x <= 2 else 'REAL')
    else:
        # Convert string labels with better error handling
        df['actual_label'] = df[label_col].str.upper().map(label_mapping)
        
        # Handle any unmapped values
        unmapped_mask = df['actual_label'].isna()
        if unmapped_mask.any():
            print(f"Warning: Found {unmapped_mask.sum()} unmapped labels: {df[label_col][unmapped_mask].unique()}")
            # Remove rows with unmapped labels
            df = df[~unmapped_mask]
    
    # Final check for NaN values
    if df['actual_label'].isna().any():
        print("Error: Still found NaN values in labels after cleaning")
        print("Label values found:", df[label_col].unique())
        raise ValueError("Dataset contains invalid labels that cannot be mapped")
    
    print(f"Validation dataset loaded: {len(df)} reviews")
    print(f"Label distribution: {df['actual_label'].value_counts().to_dict()}")
    print(f"Unique label values in original data: {df[label_col].unique()}")
    
    return df

def evaluate_model(df, model_path="./fine_tuned_model"):
    """Evaluate the model on the dataset"""
    
    # Load model
    classifier = load_fake_review_model()
    
    # Predict on all data
    predictions = []
    confidences = []
    
    print("Making predictions...")
    total = len(df)
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing review {i+1}/{len(df)}")
        if i % max(1, total // 20) == 0:
            percent = int(100 * i / total)
            update_progress(PROGRESS_VALIDATE_FILE, percent, f'Validating... ({i+1}/{total})')
        
        result = classifier(row['text'][:512])[0]
        predictions.append(result['label'])
        confidences.append(result['score'])
    
    # Map model predictions to our expected labels
    # Fine-tuned models often use LABEL_0, LABEL_1 instead of meaningful names
    mapped_predictions = []
    for pred in predictions:
        if pred in ['LABEL_0', '0', 'REAL']:
            mapped_predictions.append('REAL')
        elif pred in ['LABEL_1', '1', 'FAKE']:
            mapped_predictions.append('FAKE')
        else:
            # If it's an unknown label, try to infer from the model's label mapping
            print(f"Warning: Unknown prediction label '{pred}', treating as FAKE")
            mapped_predictions.append('FAKE')
    
    # Add predictions to dataframe
    df['predicted_label'] = mapped_predictions
    df['confidence'] = confidences
    
    # Debug: Check what labels the model is predicting
    print(f"Original model predictions: {set(predictions)}")
    print(f"Mapped predictions: {set(mapped_predictions)}")
    print(f"Actual labels: {set(df['actual_label'])}")
    
    return df

def create_evaluation_report(df):
    """Create comprehensive evaluation report"""
    
    # Calculate metrics
    y_true = df['actual_label']
    y_pred = df['predicted_label']
    
    # Check if we have binary classification
    unique_labels = set(y_true) | set(y_pred)
    print(f"Unique labels found: {unique_labels}")
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle binary vs multiclass
    if len(unique_labels) == 2:
        # Binary classification
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label='FAKE')
    else:
        # Multiclass classification - use weighted average
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        print("Warning: Multiclass classification detected, using weighted averaging")
    
    # Ensure we only have REAL and FAKE labels for confusion matrix
    valid_labels = ['REAL', 'FAKE']
    y_true_clean = [label if label in valid_labels else 'REAL' for label in y_true]
    y_pred_clean = [label if label in valid_labels else 'REAL' for label in y_pred]
    
    # Confusion matrix
    try:
        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=valid_labels)
        
        print("=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)
        print(f"Total Reviews: {len(df)}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
        print(f"Recall: {recall:.3f} ({recall*100:.1f}%)")
        print(f"F1-Score: {f1:.3f} ({f1*100:.1f}%)")
        print()
        
        print("Confusion Matrix:")
        print("                 Predicted")
        print("                 REAL  FAKE")
        print(f"Actual REAL     {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       FAKE     {cm[1][0]:4d}  {cm[1][1]:4d}")
        print()
        
        # Detailed classification report
        print("Detailed Classification Report:")
        print(classification_report(y_true_clean, y_pred_clean, target_names=valid_labels))
        
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        print("Skipping confusion matrix and detailed report...")
    
    # Confidence analysis
    print("Confidence Analysis:")
    fake_reviews = df[df['actual_label'] == 'FAKE']
    real_reviews = df[df['actual_label'] == 'REAL']
    
    print(f"Average confidence for FAKE reviews: {fake_reviews['confidence'].mean():.3f}")
    print(f"Average confidence for REAL reviews: {real_reviews['confidence'].mean():.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def show_sample_predictions(df, n_samples=10):
    """Show sample predictions for demonstration"""
    
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Show some correct predictions
    correct_predictions = df[df['actual_label'] == df['predicted_label']].head(n_samples//2)
    incorrect_predictions = df[df['actual_label'] != df['predicted_label']].head(n_samples//2)
    
    print("\nCORRECT PREDICTIONS:")
    for _, row in correct_predictions.iterrows():
        print(f"Text: {row['text'][:80]}...")
        print(f"Actual: {row['actual_label']}, Predicted: {row['predicted_label']} (conf: {row['confidence']:.3f})")
        print("-" * 40)
    
    print("\nINCORRECT PREDICTIONS:")
    for _, row in incorrect_predictions.iterrows():
        print(f"Text: {row['text'][:80]}...")
        print(f"Actual: {row['actual_label']}, Predicted: {row['predicted_label']} (conf: {row['confidence']:.3f})")
        print("-" * 40)

def create_validation_set(df, test_size=0.2):
    """Create a validation set for robust evaluation"""
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['actual_label'])
    
    print(f"Training set: {len(train_df)} reviews")
    print(f"Validation set: {len(val_df)} reviews")
    
    # Save validation set in root folder
    validation_path = os.path.join("..", "validation_set.csv")
    val_df.to_csv(validation_path, index=False)
    print(f"Validation set saved to {validation_path}")
    
    return train_df, val_df

# ============================================================================
# TESTING AND DEMO
# ============================================================================

def test_fine_tuned_model(test_texts):
    """Test the fine-tuned model on sample texts"""
    
    # Load the fine-tuned model
    classifier = load_fake_review_model()
    
    results = []
    for text in test_texts:
        result = classifier(text[:512])[0]
        results.append({
            "text": text,
            "prediction": result['label'],
            "confidence": result['score']
        })
    
    return results

def run_validation_pipeline(csv_file_path, start_range=None, end_range=None):
    """Complete validation pipeline"""
    
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found!")
        return None
    
    print("Loading dataset...")
    df = load_data_for_validation(csv_file_path, start_range, end_range)
    
    # Create validation set
    train_df, val_df = create_validation_set(df)
    
    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_df_with_predictions = evaluate_model(val_df)
    
    # Create evaluation report
    metrics = create_evaluation_report(val_df_with_predictions)
    
    # Show sample predictions
    show_sample_predictions(val_df_with_predictions)
    
    print("\nValidation completed! Check the results above.")
    return metrics

# ============================================================================
# DATASET UTILITIES
# ============================================================================

def get_dataset_info(csv_file_path):
    """Get information about a dataset"""
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found!")
        return None
    
    try:
        df = pd.read_csv(csv_file_path)
        total_reviews = len(df)
        
        print(f"Dataset: {csv_file_path}")
        print(f"Total records: {total_reviews:,}")
        print(f"Columns: {list(df.columns)}")
        
        return {
            'total_reviews': total_reviews,
            'columns': list(df.columns),
            'file_path': csv_file_path
        }
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None

def get_user_range_selection(total_reviews):
    """Get range selection from user"""
    print(f"\nDataset has {total_reviews:,} total records.")
    print("Enter the range of records to use (e.g., 0-1000 for first 1000 records)")
    
    while True:
        try:
            range_input = input("Enter range (start-end) or 'all' for entire dataset: ").strip()
            
            if range_input.lower() == 'all':
                return 0, total_reviews
            
            if '-' in range_input:
                start, end = map(int, range_input.split('-'))
                if 0 <= start < end <= total_reviews:
                    return start, end
                else:
                    print(f"Invalid range. Must be between 0 and {total_reviews}")
            else:
                print("Invalid format. Use 'start-end' (e.g., 0-1000) or 'all'")
                
        except ValueError:
            print("Invalid input. Please enter numbers separated by '-'")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None, None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if labeled dataset exists
    csv_file = "labeled_fake_reviews.csv"  # Update this to your actual file name
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Please make sure your labeled dataset is in the current directory.")
        exit(1)
    
    # Fine-tune the model
    print("Starting fine-tuning process...")
    results = fine_tune_model(csv_file)
    
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
    
    # Run validation
    print("\nRunning validation...")
    run_validation_pipeline(csv_file) 