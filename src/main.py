from fastapi import FastAPI, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from transformers import pipeline
import re
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from datetime import datetime
import pandas as pd
import json
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend assets
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend")), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))

@app.get("/batch")
def read_batch():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "batch.html"))

@app.get("/accuracy")
def read_accuracy():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "accuracy.html"))

@app.get("/about")
def read_about():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "about.html"))

@app.get("/fine_tune")
def read_fine_tune():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "fine_tune.html"))

@app.get("/styles.css")
def get_styles():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "styles.css"))

class ReviewInput(BaseModel):
    review_text: str
    username: str
    num_reviews: Optional[int] = None
    timestamp: Optional[str] = None
    rating: Optional[int] = None  # Add rating for behavioral analysis

class ReviewOutput(BaseModel):
    fraud_score: float
    is_flagged: bool
    reason: str
    ai_score: Optional[float] = None
    behavioral_score: Optional[float] = None
    heuristic_score: Optional[float] = None
    recommended_action: Optional[str] = None

class LabeledReviewInput(BaseModel):
    review_text: str
    username: str
    actual_label: str  # "FAKE" or "REAL"
    num_reviews: Optional[int] = None
    timestamp: Optional[str] = None

class AccuracyResult(BaseModel):
    total_reviews: int
    correct_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Dict[str, int]
    detailed_results: List[Dict[str, Any]]

# Load sentiment analysis pipeline (using a small model for demo)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load the fake review detection model (use fine-tuned if available)
import os

# Check for fine-tuned model in both current directory and parent directory
fine_tuned_path = None
if os.path.exists("./fine_tuned_model"):
    fine_tuned_path = "./fine_tuned_model"
elif os.path.exists("../fine_tuned_model"):
    fine_tuned_path = "../fine_tuned_model"

if fine_tuned_path:
    print(f"Loading fine-tuned model from {fine_tuned_path}...")
    fake_review_detector = pipeline(
        "text-classification",
        model=fine_tuned_path
    )
else:
    print("Loading pre-trained model...")
    fake_review_detector = pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-fake-news-detection"
    )

def detect_suspicious_text(text):
    # More refined heuristics with better thresholds
    if text.count("!") > 5:  # Increased from 3 to 5
        return True, "Excessive exclamation marks"
    
    # More sophisticated word repetition detection
    words = text.lower().split()
    word_counts = {}
    for word in words:
        if len(word) > 3:  # Only count words longer than 3 characters
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Check for excessive repetition of specific words
    repeated_words = [word for word, count in word_counts.items() if count > 3]
    if len(repeated_words) > 1:  # Only flag if multiple words are repeated
        return True, f"Excessive repetition of words: {', '.join(repeated_words[:3])}"
    
    if text.isupper() and len(text) > 20:  # Only flag if longer than 20 chars
        return True, "All caps text"
    
    # Add more sophisticated checks
    if len(re.findall(r'\b[A-Z]{3,}\b', text)) > 2:  # Multiple ALL CAPS words
        return True, "Multiple all-caps words"
    if text.count("?") > 3:  # Too many questions
        return True, "Excessive questions"
    
    # Check for suspicious patterns
    if text.count("great") > 2 or text.count("amazing") > 2 or text.count("excellent") > 2:
        return True, "Excessive positive adjectives"
    
    # Check for overly generic language
    generic_phrases = ["great product", "highly recommend", "best ever", "love it", "perfect"]
    generic_count = sum(1 for phrase in generic_phrases if phrase in text.lower())
    if generic_count > 2:
        return True, "Overly generic language"
    
    # Check for lack of specific details
    if len(text.split()) > 20 and len(set(text.lower().split())) < len(text.split()) * 0.6:
        return True, "Low vocabulary diversity"
    
    return False, ""

def analyze_review_complexity(text):
    """Analyze review complexity and return a score"""
    words = text.split()
    if len(words) < 5:
        return 0.2  # Very simple
    
    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Calculate unique word ratio
    unique_ratio = len(set(words)) / len(words)
    
    # Calculate sentence complexity (simple heuristic)
    sentences = text.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
    
    # Combine factors
    complexity_score = (avg_word_length * 0.3 + unique_ratio * 0.4 + min(avg_sentence_length / 10, 1) * 0.3)
    return min(complexity_score, 1.0)

def is_fake_review(text):
    truncated = text[:512]
    result = fake_review_detector(truncated)[0]
    label = result['label']  # Could be 'FAKE'/'REAL' or 'LABEL_0'/'LABEL_1'
    score = result['score']
    
    # Debug: Print first few predictions to see what labels we're getting
    if hasattr(is_fake_review, '_debug_count'):
        is_fake_review._debug_count += 1
    else:
        is_fake_review._debug_count = 1
    
    if is_fake_review._debug_count <= 5:
        print(f"DEBUG: Model prediction - label: '{label}', score: {score:.3f}")
        # Also check if this is a fine-tuned model
        if is_fake_review._debug_count == 1:
            using_fine_tuned = os.path.exists("./fine_tuned_model") or os.path.exists("../fine_tuned_model")
            print(f"DEBUG: Model type check - using fine-tuned model: {using_fine_tuned}")
            if score > 0.95:
                print(f"WARNING: Model showing very high confidence ({score:.3f}) - may be overconfident")
    
    # Handle fine-tuned model labels (LABEL_0, LABEL_1)
    if label in ['LABEL_0', '0', 'REAL']:
        return False, score  # Not fake
    elif label in ['LABEL_1', '1', 'FAKE']:
        return True, score   # Is fake
    else:
        # Fallback: assume high confidence means fake
        print(f"Warning: Unknown label '{label}', treating as fake if confidence > 0.5")
        return score > 0.5, score

@app.post("/analyze_review", response_model=ReviewOutput)
def analyze_review(input: ReviewInput):
    # Import behavioral analyzer
    from behavioral_analyzer import behavioral_analyzer
    
    # Truncate text for model compatibility
    truncated_text = input.review_text[:512]
    
    # Get fake review model prediction (primary decision maker)
    is_fake, fake_conf = is_fake_review(truncated_text)
    
    # Get sentiment for additional context
    sentiment = sentiment_analyzer(truncated_text)[0]
    
    # MULTI-SIGNAL APPROACH: Combine AI model with behavioral analysis
    fraud_score = 0
    reasons = []
    
    # 1. AI Model Score (adjusted weight based on model type)
    if is_fake:
        # Cap the AI score to prevent overconfident predictions
        ai_score = min(fake_conf, 0.85)  # Cap at 85% confidence
        if fake_conf > 0.8:
            reasons.append(f"AI: High confidence fake review ({fake_conf:.2f})")
        elif fake_conf > 0.6:
            reasons.append(f"AI: Likely fake review ({fake_conf:.2f})")
        else:
            reasons.append(f"AI: Possibly fake review ({fake_conf:.2f})")
    else:
        ai_score = 1 - fake_conf
        if fake_conf > 0.8:
            reasons.append(f"AI: High confidence genuine review ({fake_conf:.2f})")
        else:
            reasons.append(f"AI: Likely genuine review ({fake_conf:.2f})")
    
    # 2. Behavioral Analysis (30% weight)
    behavioral_result = behavioral_analyzer.analyze_user_behavior(
        username=input.username,
        num_reviews=input.num_reviews,
        timestamp=input.timestamp,
        rating=input.rating,
        review_text=input.review_text
    )
    
    behavioral_score = behavioral_result['risk_score']
    if behavioral_result['reasons']:
        reasons.extend([f"Behavior: {reason}" for reason in behavioral_result['reasons']])
    
    # 3. Text Heuristics (10% weight)
    heuristic_score = 0
    suspicious, reason_text = detect_suspicious_text(input.review_text)
    if suspicious:
        heuristic_score = 0.3
        reasons.append(f"Text: {reason_text}")
    
    if len(input.review_text) < 10:
        heuristic_score = max(heuristic_score, 0.2)
        reasons.append("Text: Very short review")
    
    # Combine all signals with weights - adjusted for fine-tuned model bias
    if os.path.exists("./fine_tuned_model") or os.path.exists("../fine_tuned_model"):
        # Check if we have behavioral data (for accuracy testing vs real analysis)
        has_behavioral_data = (input.num_reviews is not None or input.timestamp is not None or input.rating is not None)
        
        if has_behavioral_data:
            # Full analysis with behavioral data
            fraud_score = (
                ai_score * 0.4 +           # AI model (40%)
                behavioral_score * 0.4 +   # Behavioral analysis (40%)
                heuristic_score * 0.2      # Text heuristics (20%)
            )
        else:
            # Accuracy testing - rely more on AI model since no behavioral data
            fraud_score = (
                ai_score * 0.7 +           # AI model (70% - increased for testing)
                behavioral_score * 0.1 +   # Behavioral analysis (10% - minimal)
                heuristic_score * 0.2      # Text heuristics (20%)
            )
    else:
        # Use original weights for pre-trained model
        fraud_score = (
            ai_score * 0.6 +           # AI model (60%)
            behavioral_score * 0.3 +   # Behavioral analysis (30%)
            heuristic_score * 0.1      # Text heuristics (10%)
        )
    
    # Ensure score is between 0 and 1
    fraud_score = max(0.0, min(1.0, fraud_score))
    
    # Debug: Print fraud score for first few reviews
    if hasattr(analyze_review, '_debug_count'):
        analyze_review._debug_count += 1
    else:
        analyze_review._debug_count = 1
    
    if analyze_review._debug_count <= 3:
        print(f"DEBUG: Fraud score breakdown - AI: {ai_score:.3f}, Behavioral: {behavioral_score:.3f}, Heuristic: {heuristic_score:.3f}, Final: {fraud_score:.3f}")
    
    # Decision threshold - adjusted for better balance
    # If using fine-tuned model, use balanced threshold
    if os.path.exists("./fine_tuned_model") or os.path.exists("../fine_tuned_model"):
        is_flagged = fraud_score > 0.45  # Lower threshold for better recall
    else:
        is_flagged = fraud_score > 0.5  # Standard threshold for pre-trained model

    # --- Recommended Action Logic ---
    if fraud_score > 0.9:
        recommended_action = "Block and Report"
    elif fraud_score > 0.7:
        recommended_action = "Flag for Human Review"
    else:
        recommended_action = "Approve"
    # --- End Recommended Action Logic ---

    if not reasons:
        reasons.append("No strong fraud signals detected.")

    return ReviewOutput(
        fraud_score=fraud_score, 
        is_flagged=is_flagged, 
        reason=", ".join(reasons),
        ai_score=ai_score,
        behavioral_score=behavioral_score,
        heuristic_score=heuristic_score,
        recommended_action=recommended_action
    )

@app.post("/analyze_reviews_batch")
def analyze_reviews_batch(inputs: List[ReviewInput]):
    results = []
    for input in inputs:
        # Use the same logic as analyze_review function
        prediction = analyze_review(input)
        results.append({
            "fraud_score": prediction.fraud_score,
            "is_flagged": prediction.is_flagged,
            "reason": prediction.reason,
            "ai_score": prediction.ai_score,
            "behavioral_score": prediction.behavioral_score,
            "heuristic_score": prediction.heuristic_score,
            "recommended_action": getattr(prediction, "recommended_action", None)
        })
    return results

@app.post("/get_csv_info")
def get_csv_info(file: UploadFile = File(...)):
    # Read CSV file to get total number of reviews
    df = pd.read_csv(file.file)
    total_reviews = len(df)
    return {"total_reviews": total_reviews}

@app.post("/analyze_reviews_csv")
def analyze_reviews_csv(file: UploadFile = File(...), start: int = Form(0), end: int = Form(100)):
    # Read CSV file into pandas DataFrame
    df = pd.read_csv(file.file)
    print(f"DEBUG: Original CSV has {len(df)} rows")
    print(f"DEBUG: Requested range: {start} to {end}")
    # Apply range limits
    df = df.iloc[start:end]
    print(f"DEBUG: After slicing, DataFrame has {len(df)} rows")
    reviews = []
    for _, row in df.iterrows():
        # Try to extract fields from common column names
        review_text = row.get('reviewText') or row.get('Text') or row.get('review') or ''
        username = row.get('UserId') or row.get('ProfileName') or row.get('username') or 'anonymous'
        num_reviews = None
        timestamp = row.get('Time') or None
        rating = row.get('Score') or row.get('rating') or None
        
        # Convert rating to int if it exists
        if rating is not None:
            try:
                rating = int(rating)
            except:
                rating = None
        
        # Convert Unix time to ISO if needed
        if pd.notnull(timestamp):
            try:
                timestamp = str(int(timestamp))
                if len(timestamp) > 10:
                    timestamp = timestamp[:10]
                timestamp = datetime.utcfromtimestamp(int(timestamp)).isoformat()
            except Exception:
                timestamp = None
        reviews.append(ReviewInput(
            review_text=review_text,
            username=username,
            num_reviews=num_reviews,
            timestamp=timestamp,
            rating=rating
        ))
    return analyze_reviews_batch(reviews) 

@app.post("/calculate_accuracy", response_model=AccuracyResult)
def calculate_accuracy(labeled_reviews: List[LabeledReviewInput]):
    """
    Calculate accuracy of the fake review detection system against labeled data.
    """
    total_reviews = len(labeled_reviews)
    correct_predictions = 0
    true_positives = 0  # Predicted fake, actually fake
    false_positives = 0  # Predicted fake, actually real
    true_negatives = 0  # Predicted real, actually real
    false_negatives = 0  # Predicted real, actually fake
    
    detailed_results = []
    
    # Track statistics for debugging
    fake_model_correct = 0
    fake_model_total = 0
    
    for i, labeled_review in enumerate(labeled_reviews):
        # Convert to regular ReviewInput for analysis
        review_input = ReviewInput(
            review_text=labeled_review.review_text,
            username=labeled_review.username,
            num_reviews=labeled_review.num_reviews,
            timestamp=labeled_review.timestamp
        )
        
        # Get system prediction
        prediction = analyze_review(review_input)
        
        # Determine predicted label
        predicted_label = "FAKE" if prediction.is_flagged else "REAL"
        actual_label = labeled_review.actual_label.upper()
        
        # Track fake model accuracy
        is_fake, fake_conf = is_fake_review(labeled_review.review_text[:512])
        fake_model_total += 1
        if (is_fake and actual_label == "FAKE") or (not is_fake and actual_label == "REAL"):
            fake_model_correct += 1
        
        # Update counters
        if predicted_label == actual_label:
            correct_predictions += 1
        
        if predicted_label == "FAKE" and actual_label == "FAKE":
            true_positives += 1
        elif predicted_label == "FAKE" and actual_label == "REAL":
            false_positives += 1
        elif predicted_label == "REAL" and actual_label == "REAL":
            true_negatives += 1
        elif predicted_label == "REAL" and actual_label == "FAKE":
            false_negatives += 1
        
        # Store detailed result
        detailed_results.append({
            "review_number": i + 1,
            "review_text": labeled_review.review_text[:100] + "..." if len(labeled_review.review_text) > 100 else labeled_review.review_text,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "fraud_score": prediction.fraud_score,
            "is_correct": predicted_label == actual_label,
            "reason": prediction.reason,
            "fake_model_prediction": "FAKE" if is_fake else "REAL",
            "fake_model_confidence": fake_conf
        })
    
    # Calculate metrics
    accuracy = correct_predictions / total_reviews if total_reviews > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    confusion_matrix = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives
    }
    
    # Add debugging info
    debug_info = {
        "fake_model_accuracy": fake_model_correct / fake_model_total if fake_model_total > 0 else 0,
        "fake_model_correct": fake_model_correct,
        "fake_model_total": fake_model_total,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives
    }
    
    # Print debugging info to console
    print(f"DEBUG: Fake model accuracy: {debug_info['fake_model_accuracy']:.2f}")
    print(f"DEBUG: True positives: {true_positives}, False positives: {false_positives}")
    print(f"DEBUG: True negatives: {true_negatives}, False negatives: {false_negatives}")
    print(f"DEBUG: Precision: {precision:.2f}, Recall: {recall:.2f}")
    print(f"DEBUG: Total reviews processed: {total_reviews}")
    print(f"DEBUG: Correct predictions: {correct_predictions}")
    print(f"DEBUG: Overall accuracy: {accuracy:.2f}")
    
    # Additional debugging for label distribution
    actual_labels = [r.actual_label for r in labeled_reviews]
    predicted_labels = ["FAKE" if r.is_flagged else "REAL" for r in [analyze_review(ReviewInput(
        review_text=r.review_text,
        username=r.username,
        num_reviews=r.num_reviews,
        timestamp=r.timestamp
    )) for r in labeled_reviews]]
    
    print(f"DEBUG: Actual label distribution: {dict(zip(*np.unique(actual_labels, return_counts=True)))}")
    print(f"DEBUG: Predicted label distribution: {dict(zip(*np.unique(predicted_labels, return_counts=True)))}")
    
    return AccuracyResult(
        total_reviews=total_reviews,
        correct_predictions=correct_predictions,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        confusion_matrix=confusion_matrix,
        detailed_results=detailed_results
    )

@app.get("/test_fake_model")
def test_fake_model():
    """Test endpoint to verify fake review model is working"""
    test_texts = [
        "This product is absolutely amazing! Best purchase ever!",
        "The quality is terrible and I hate it completely.",
        "This is a computer generated fake review for testing purposes.",
        "I bought this item and it works as expected. Good value for money."
    ]
    
    results = []
    for text in test_texts:
        is_fake, conf = is_fake_review(text[:512])
        results.append({
            "text": text,
            "is_fake": is_fake,
            "confidence": conf,
            "prediction": "FAKE" if is_fake else "REAL"
        })
    
    # Add detailed model info
    model_info = {
        "using_fine_tuned": os.path.exists("./fine_tuned_model"),
        "model_path": "./fine_tuned_model" if os.path.exists("./fine_tuned_model") else "pre-trained"
    }
    
    return {"test_results": results, "model_info": model_info}

@app.post("/get_labeled_csv_info")
def get_labeled_csv_info(file: UploadFile = File(...)):
    # Read CSV file to get total number of reviews
    df = pd.read_csv(file.file)
    total_reviews = len(df)
    return {"total_reviews": total_reviews}

@app.post("/analyze_labeled_csv")
def analyze_labeled_csv(file: UploadFile = File(...), start: int = Form(0), end: int = Form(100)):
    """
    Analyze a CSV file with labeled fake/real reviews and calculate accuracy.
    Expected columns: category, rating, label (OR/CG), text
    """
    df = pd.read_csv(file.file)
    print(f"DEBUG: Original labeled CSV has {len(df)} rows")
    print(f"DEBUG: Requested range: {start} to {end}")
    
    # Apply range limits
    df = df.iloc[start:end]
    print(f"DEBUG: After slicing, labeled DataFrame has {len(df)} rows")
    
    # Use the same robust column detection as script runner
    columns = df.columns.tolist()
    print(f"DEBUG: Detected columns: {columns}")
    
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
        print(f"ERROR: Could not detect text and label columns. Available columns: {columns}")
        return {"error": f"Could not detect text and label columns. Available columns: {columns}"}
    
    print(f"DEBUG: Mapped columns: text='{text_col}', label='{label_col}'")
    
    # Clean and prepare text data
    df['text'] = df[text_col].fillna('').astype(str)
    df = df[df['text'].str.len() > 10]
    df = df[df['text'].str.len() < 1000]
    
    # Map labels for validation with better error handling (same as script runner)
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
            print(f"WARNING: Found {unmapped_mask.sum()} unmapped labels: {df[label_col][unmapped_mask].unique()}")
            # Remove rows with unmapped labels
            df = df[~unmapped_mask]
    
    # Final check for NaN values
    if df['actual_label'].isna().any():
        print("ERROR: Still found NaN values in labels after cleaning")
        print("Label values found:", df[label_col].unique())
        return {"error": "Dataset contains invalid labels that cannot be mapped"}
    
    print(f"DEBUG: Validation dataset loaded: {len(df)} reviews")
    print(f"DEBUG: Label distribution: {df['actual_label'].value_counts().to_dict()}")
    print(f"DEBUG: Unique label values in original data: {df[label_col].unique()}")
    
    # Convert DataFrame to labeled reviews
    labeled_reviews = []
    for _, row in df.iterrows():
        labeled_reviews.append(LabeledReviewInput(
            review_text=row['text'],
            username=row.get('category', 'unknown'),
            actual_label=row['actual_label'],
            num_reviews=None,
            timestamp=None
        ))
    
    # Calculate accuracy
    return calculate_accuracy(labeled_reviews) 

@app.get("/fine_tune_progress")
def fine_tune_progress():
    progress_file = os.path.join(os.path.dirname(__file__), '..', 'progress_fine_tune.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    else:
        return JSONResponse(content={"percent": 0, "message": "Not started."})

@app.get("/validate_progress")
def validate_progress():
    progress_file = os.path.join(os.path.dirname(__file__), '..', 'progress_validate.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    else:
        return JSONResponse(content={"percent": 0, "message": "Not started."}) 