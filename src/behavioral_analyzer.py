"""
Behavioral Analysis Module for Fake Review Detection
Analyzes user behavior patterns, timing, and other non-textual signals
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

class BehavioralAnalyzer:
    def __init__(self):
        # Store user behavior data (in a real system, this would be a database)
        self.user_history = {}
        self.review_timestamps = {}
    
    def analyze_user_behavior(self, username: str, num_reviews: Optional[int] = None, 
                            timestamp: Optional[str] = None, rating: Optional[int] = None,
                            review_text: str = "") -> Dict[str, Any]:
        """
        Analyze user behavior patterns for fraud detection
        """
        fraud_signals = []
        risk_score = 0.0
        reasons = []
        
        # Initialize user history if not exists
        if username not in self.user_history:
            self.user_history[username] = {
                'reviews': [],
                'first_review_time': timestamp,
                'total_reviews': 0,
                'ratings': [],
                'review_texts': []
            }
        
        user_data = self.user_history[username]
        
        # 1. New User Analysis
        if num_reviews is not None and num_reviews < 3:
            fraud_signals.append("new_user_few_reviews")
            risk_score += 0.2
            reasons.append(f"New user with only {num_reviews} reviews")
        
        # 2. Review Frequency Analysis
        if timestamp:
            recent_reviews = self._get_recent_reviews(username, timestamp, hours=24)
            if len(recent_reviews) > 5:
                fraud_signals.append("rapid_posting")
                risk_score += 0.3
                reasons.append(f"Posted {len(recent_reviews)} reviews in 24 hours")
        
        # 3. Rating Pattern Analysis
        if rating is not None:
            user_data['ratings'].append(rating)
            if len(user_data['ratings']) >= 3:
                rating_pattern = self._analyze_rating_pattern(user_data['ratings'])
                if rating_pattern['only_extreme']:
                    fraud_signals.append("extreme_ratings_only")
                    risk_score += 0.25
                    reasons.append("Only posts extreme ratings (1 or 5 stars)")
        
        # 4. Review Text Similarity Analysis
        if review_text:
            user_data['review_texts'].append(review_text)
            if len(user_data['review_texts']) >= 2:
                similarity_score = self._check_text_similarity(user_data['review_texts'])
                if similarity_score > 0.8:
                    fraud_signals.append("duplicate_reviews")
                    risk_score += 0.4
                    reasons.append("Very similar review texts detected")
        
        # 5. Unusual Timing Analysis
        if timestamp:
            unusual_time = self._check_unusual_timing(timestamp)
            if unusual_time:
                fraud_signals.append("unusual_timing")
                risk_score += 0.15
                reasons.append("Posted at unusual time of day")
        
        # 6. Account Age Analysis
        if user_data['first_review_time'] and timestamp:
            account_age = self._calculate_account_age(user_data['first_review_time'], timestamp)
            if account_age < 1:  # Less than 1 day
                fraud_signals.append("very_new_account")
                risk_score += 0.2
                reasons.append("Very new account (less than 1 day old)")
        
        # Update user history
        user_data['total_reviews'] += 1
        user_data['reviews'].append({
            'timestamp': timestamp,
            'rating': rating,
            'text': review_text
        })
        
        # Cap risk score at 1.0
        risk_score = min(risk_score, 1.0)
        
        return {
            'fraud_signals': fraud_signals,
            'risk_score': risk_score,
            'reasons': reasons,
            'user_data': {
                'total_reviews': user_data['total_reviews'],
                'account_age_days': self._calculate_account_age(user_data['first_review_time'], timestamp) if timestamp else None
            }
        }
    
    def _get_recent_reviews(self, username: str, current_time: str, hours: int = 24) -> List[Dict]:
        """Get recent reviews for a user within specified hours"""
        if username not in self.user_history:
            return []
        
        try:
            current_dt = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
            cutoff_time = current_dt - timedelta(hours=hours)
            
            recent_reviews = []
            for review in self.user_history[username]['reviews']:
                if review['timestamp']:
                    review_dt = datetime.fromisoformat(review['timestamp'].replace('Z', '+00:00'))
                    if review_dt >= cutoff_time:
                        recent_reviews.append(review)
            
            return recent_reviews
        except:
            return []
    
    def _analyze_rating_pattern(self, ratings: List[int]) -> Dict[str, Any]:
        """Analyze rating patterns for suspicious behavior"""
        if not ratings:
            return {'only_extreme': False, 'avg_rating': 0}
        
        extreme_ratings = [r for r in ratings if r in [1, 5]]
        avg_rating = sum(ratings) / len(ratings)
        
        return {
            'only_extreme': len(extreme_ratings) == len(ratings),
            'avg_rating': avg_rating,
            'extreme_ratio': len(extreme_ratings) / len(ratings)
        }
    
    def _check_text_similarity(self, texts: List[str]) -> float:
        """Check similarity between review texts"""
        if len(texts) < 2:
            return 0.0
        
        # Simple similarity check using common words
        words1 = set(re.findall(r'\b\w+\b', texts[-1].lower()))
        words2 = set(re.findall(r'\b\w+\b', texts[-2].lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _check_unusual_timing(self, timestamp: str) -> bool:
        """Check if review was posted at unusual time"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            
            # Unusual times: 2 AM - 6 AM
            return 2 <= hour <= 6
        except:
            return False
    
    def _calculate_account_age(self, first_review: str, current_review: str) -> float:
        """Calculate account age in days"""
        try:
            first_dt = datetime.fromisoformat(first_review.replace('Z', '+00:00'))
            current_dt = datetime.fromisoformat(current_review.replace('Z', '+00:00'))
            
            delta = current_dt - first_dt
            return delta.days
        except:
            return 0.0

# Global instance
behavioral_analyzer = BehavioralAnalyzer() 