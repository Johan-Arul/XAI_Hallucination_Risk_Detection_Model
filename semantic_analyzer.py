"""
Semantic analysis module for detecting inconsistencies in legal text.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re


class SemanticAnalyzer:
    """Analyzes semantic consistency and coherence in legal documents."""
    
    def __init__(self, config: Dict):
        """Initialize the semantic analyzer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        
        self.stop_words = set(stopwords.words('english'))
        
        # Load legal-specific terms and patterns
        self.legal_terms = self._load_legal_terminology()
        self.inconsistency_patterns = self._load_inconsistency_patterns()
    
    def _load_legal_terminology(self) -> Dict[str, List[str]]:
        """Load legal terminology for better analysis."""
        return {
            "contract_terms": [
                "whereas", "heretofore", "hereinafter", "notwithstanding",
                "covenant", "warranty", "indemnification", "severability"
            ],
            "procedural_terms": [
                "plaintiff", "defendant", "motion", "discovery", "deposition",
                "subpoena", "jurisdiction", "venue", "standing"
            ],
            "citation_markers": [
                "supra", "infra", "see", "cf.", "accord", "but see",
                "compare", "contra", "see generally"
            ]
        }
    
    def _load_inconsistency_patterns(self) -> List[Dict]:
        """Load patterns that indicate potential inconsistencies."""
        return [
            {
                "pattern": r"(\d{4})\s+.*\s+(\d{4})",
                "type": "date_inconsistency",
                "description": "Potential conflicting dates"
            },
            {
                "pattern": r"(shall|must|will)\s+not.*\s+(shall|must|will)",
                "type": "obligation_conflict",
                "description": "Conflicting obligation statements"
            },
            {
                "pattern": r"(plaintiff|defendant).*\s+(plaintiff|defendant)",
                "type": "party_confusion",
                "description": "Potential party role confusion"
            }
        ]
    
    def analyze_consistency(self, text: str) -> float:
        """
        Analyze semantic consistency of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Consistency score between 0 and 1
        """
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                return 1.0  # Single sentence is consistent by default
            
            # Calculate sentence embeddings
            embeddings = self.embedding_model.encode(sentences)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings)
            
            # Calculate consistency metrics
            coherence_score = self._calculate_coherence(similarities)
            terminology_score = self._analyze_terminology_consistency(sentences)
            logical_score = self._detect_logical_inconsistencies(text)
            
            # Weighted combination
            weights = {
                "coherence": 0.4,
                "terminology": 0.3,
                "logical": 0.3
            }
            
            consistency_score = (
                coherence_score * weights["coherence"] +
                terminology_score * weights["terminology"] +
                logical_score * weights["logical"]
            )
            
            return consistency_score
            
        except Exception as e:
            self.logger.error(f"Error in consistency analysis: {e}")
            return 0.5  # Return neutral score on error
    
    def _calculate_coherence(self, similarity_matrix: np.ndarray) -> float:
        """Calculate coherence score from similarity matrix."""
        if similarity_matrix.shape[0] < 2:
            return 1.0
        
        # Remove diagonal (self-similarity)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[mask]
        
        # Calculate average similarity
        mean_similarity = np.mean(similarities)
        
        # Penalize very low similarities (potential inconsistencies)
        low_similarity_penalty = np.sum(similarities < 0.3) / len(similarities)
        
        coherence = mean_similarity - (low_similarity_penalty * 0.2)
        return max(0.0, min(1.0, coherence))
    
    def _analyze_terminology_consistency(self, sentences: List[str]) -> float:
        """Analyze consistency of legal terminology usage."""
        term_usage = {}
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for category, terms in self.legal_terms.items():
                for term in terms:
                    if term in ' '.join(words):
                        if term not in term_usage:
                            term_usage[term] = []
                        term_usage[term].append(sentence)
        
        # Check for consistent usage patterns
        consistency_score = 1.0
        
        for term, usages in term_usage.items():
            if len(usages) > 1:
                # Check if term is used consistently
                embeddings = self.embedding_model.encode(usages)
                similarities = cosine_similarity(embeddings)
                avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
                
                if avg_similarity < 0.7:
                    consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def _detect_logical_inconsistencies(self, text: str) -> float:
        """Detect logical inconsistencies using pattern matching."""
        inconsistency_count = 0
        
        for pattern_info in self.inconsistency_patterns:
            pattern = pattern_info["pattern"]
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            if matches:
                # Analyze matches for actual inconsistencies
                for match in matches:
                    if self._is_actual_inconsistency(match, pattern_info["type"]):
                        inconsistency_count += 1
        
        # Convert to score (more inconsistencies = lower score)
        if inconsistency_count == 0:
            return 1.0
        
        # Penalize based on text length and inconsistency count
        text_length_factor = len(text) / 1000  # Normalize by text length
        penalty = min(inconsistency_count / (text_length_factor + 1), 1.0)
        
        return max(0.0, 1.0 - penalty)
    
    def _is_actual_inconsistency(self, match: Tuple, inconsistency_type: str) -> bool:
        """Determine if a pattern match represents an actual inconsistency."""
        if inconsistency_type == "date_inconsistency":
            # Check if dates are actually conflicting
            dates = [int(d) for d in match if d.isdigit()]
            if len(dates) >= 2:
                return abs(dates[0] - dates[1]) > 50  # Suspicious if dates differ by >50 years
        
        elif inconsistency_type == "obligation_conflict":
            # Simple heuristic: if both positive and negative obligations exist
            return True
        
        elif inconsistency_type == "party_confusion":
            # Check if same party is referenced with different roles
            return True
        
        return False
    
    def detect_hallucination_markers(self, text: str) -> List[Dict]:
        """Detect specific markers that often indicate hallucinations."""
        markers = []
        
        # Vague references without proper citations
        vague_patterns = [
            r"according to (recent|a|the) (study|report|case)",
            r"it is well established that",
            r"courts have consistently held",
            r"the law clearly states"
        ]
        
        for pattern in vague_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                markers.append({
                    "type": "vague_reference",
                    "text": match.group(),
                    "position": match.span(),
                    "confidence": 0.7
                })
        
        # Overly specific claims without citations
        specific_patterns = [
            r"\d+\.\d+%",  # Specific percentages
            r"\$[\d,]+",   # Specific monetary amounts
            r"exactly \d+", # Exact numbers
        ]
        
        for pattern in specific_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Check if citation follows
                following_text = text[match.end():match.end()+100]
                if not re.search(r"(\d+\s+\w+\s+\d+|\(\d{4}\))", following_text):
                    markers.append({
                        "type": "uncited_specific_claim",
                        "text": match.group(),
                        "position": match.span(),
                        "confidence": 0.8
                    })
        
        return markers
    
    def calculate_embedding_drift(self, 
                                 original_embeddings: np.ndarray, 
                                 generated_embeddings: np.ndarray) -> float:
        """Calculate drift between original and generated content embeddings."""
        if original_embeddings.shape != generated_embeddings.shape:
            self.logger.warning("Embedding shapes don't match for drift calculation")
            return 0.5
        
        similarity = cosine_similarity(
            original_embeddings.reshape(1, -1),
            generated_embeddings.reshape(1, -1)
        )[0][0]
        
        return similarity
