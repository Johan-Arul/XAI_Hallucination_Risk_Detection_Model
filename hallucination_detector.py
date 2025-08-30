"""
Main hallucination detection system for legal documents.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .semantic_analyzer import SemanticAnalyzer
from .citation_validator import CitationValidator
from .risk_scorer import RiskScorer
from .document_processor import DocumentProcessor


@dataclass
class HallucinationResult:
    """Result of hallucination detection analysis."""
    text_segment: str
    risk_score: float
    risk_category: str
    confidence: float
    issues: List[str]
    recommendations: List[str]
    citations_verified: bool
    semantic_consistency: float


@dataclass
class DocumentAnalysisResult:
    """Complete analysis result for a document."""
    document_id: str
    overall_risk_score: float
    segment_results: List[HallucinationResult]
    summary: Dict[str, any]
    timestamp: str


class LegalHallucinationDetector:
    """Main class for detecting hallucinations in legal documents."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the hallucination detector."""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.semantic_analyzer = SemanticAnalyzer(self.config)
        self.citation_validator = CitationValidator(self.config)
        self.risk_scorer = RiskScorer(self.config)
        self.document_processor = DocumentProcessor()
        
        # Load models
        self.embedding_model = self._load_embedding_model()
        
        self.logger.info("Legal Hallucination Detector initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "detection": {
                "similarity_threshold": 0.85,
                "citation_verification": True,
                "semantic_analysis": True,
                "min_segment_length": 50,
                "max_segment_length": 500
            },
            "models": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "risk_thresholds": {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.8
            }
        }
        
        if config_path:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return {**default_config, **config}
            except Exception as e:
                self.logger.warning(f"Could not load config file: {e}. Using defaults.")
        
        return default_config
    
    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the sentence embedding model."""
        model_name = self.config["models"]["embedding_model"]
        device = self.config["models"]["device"]
        
        try:
            model = SentenceTransformer(model_name, device=device)
            self.logger.info(f"Loaded embedding model: {model_name} on {device}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def detect_hallucinations(self, 
                            document_text: str, 
                            reference_sources: Optional[List[str]] = None,
                            document_id: Optional[str] = None) -> DocumentAnalysisResult:
        """
        Detect potential hallucinations in a legal document.
        
        Args:
            document_text: The legal document text to analyze
            reference_sources: Optional list of reference texts for comparison
            document_id: Optional document identifier
            
        Returns:
            DocumentAnalysisResult containing detailed analysis
        """
        if not document_id:
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting hallucination detection for document: {document_id}")
        
        # Segment the document
        segments = self._segment_document(document_text)
        
        # Analyze each segment
        segment_results = []
        for i, segment in enumerate(segments):
            self.logger.debug(f"Analyzing segment {i+1}/{len(segments)}")
            result = self._analyze_segment(segment, reference_sources)
            segment_results.append(result)
        
        # Calculate overall risk score
        overall_risk = self._calculate_overall_risk(segment_results)
        
        # Generate summary
        summary = self._generate_summary(segment_results)
        
        return DocumentAnalysisResult(
            document_id=document_id,
            overall_risk_score=overall_risk,
            segment_results=segment_results,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
    
    def _segment_document(self, text: str) -> List[str]:
        """Segment document into analyzable chunks."""
        min_length = self.config["detection"]["min_segment_length"]
        max_length = self.config["detection"]["max_segment_length"]
        
        # Simple sentence-based segmentation
        sentences = text.split('. ')
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) > max_length:
                if len(current_segment) >= min_length:
                    segments.append(current_segment.strip())
                current_segment = sentence
            else:
                current_segment += ". " + sentence if current_segment else sentence
        
        if len(current_segment) >= min_length:
            segments.append(current_segment.strip())
        
        return segments
    
    def _analyze_segment(self, 
                        segment: str, 
                        reference_sources: Optional[List[str]] = None) -> HallucinationResult:
        """Analyze a single text segment for hallucinations."""
        
        issues = []
        recommendations = []
        
        # Semantic consistency analysis
        semantic_score = self.semantic_analyzer.analyze_consistency(segment)
        if semantic_score < self.config["detection"]["similarity_threshold"]:
            issues.append("Low semantic consistency detected")
            recommendations.append("Review segment for logical consistency")
        
        # Citation verification
        citations_verified = True
        if self.config["detection"]["citation_verification"]:
            citations = self.citation_validator.extract_citations(segment)
            if citations:
                citation_results = self.citation_validator.verify_citations(citations)
                citations_verified = all(result.is_valid for result in citation_results)
                if not citations_verified:
                    issues.append("Invalid or unverifiable citations found")
                    recommendations.append("Verify all legal citations")
        
        # Calculate risk score
        risk_metrics = {
            "semantic_consistency": semantic_score,
            "citation_validity": 1.0 if citations_verified else 0.0,
            "text_length": len(segment),
            "complexity": self._calculate_complexity(segment)
        }
        
        risk_score, confidence = self.risk_scorer.calculate_risk(risk_metrics)
        risk_category = self._categorize_risk(risk_score)
        
        return HallucinationResult(
            text_segment=segment,
            risk_score=risk_score,
            risk_category=risk_category,
            confidence=confidence,
            issues=issues,
            recommendations=recommendations,
            citations_verified=citations_verified,
            semantic_consistency=semantic_score
        )
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        sentences = text.split('.')
        
        if len(sentences) == 0:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        long_words = sum(1 for word in words if len(word) > 6)
        long_word_ratio = long_words / len(words) if words else 0
        
        # Flesch-Kincaid inspired complexity
        complexity = (avg_words_per_sentence * 0.5) + (long_word_ratio * 100)
        return min(complexity / 100, 1.0)  # Normalize to 0-1
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level based on score."""
        thresholds = self.config["risk_thresholds"]
        
        if risk_score >= thresholds["high"]:
            return "HIGH"
        elif risk_score >= thresholds["medium"]:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_overall_risk(self, segment_results: List[HallucinationResult]) -> float:
        """Calculate overall document risk score."""
        if not segment_results:
            return 0.0
        
        # Weighted average with higher weight for high-risk segments
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in segment_results:
            weight = 1.0 + result.risk_score  # Higher risk gets more weight
            total_weighted_score += result.risk_score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_summary(self, segment_results: List[HallucinationResult]) -> Dict:
        """Generate analysis summary."""
        if not segment_results:
            return {}
        
        risk_categories = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        total_issues = 0
        avg_semantic_consistency = 0.0
        citation_verification_rate = 0.0
        
        for result in segment_results:
            risk_categories[result.risk_category] += 1
            total_issues += len(result.issues)
            avg_semantic_consistency += result.semantic_consistency
            citation_verification_rate += 1 if result.citations_verified else 0
        
        total_segments = len(segment_results)
        avg_semantic_consistency /= total_segments
        citation_verification_rate /= total_segments
        
        return {
            "total_segments": total_segments,
            "risk_distribution": risk_categories,
            "total_issues": total_issues,
            "average_semantic_consistency": avg_semantic_consistency,
            "citation_verification_rate": citation_verification_rate,
            "high_risk_segments": risk_categories["HIGH"],
            "recommendations": self._generate_global_recommendations(segment_results)
        }
    
    def _generate_global_recommendations(self, segment_results: List[HallucinationResult]) -> List[str]:
        """Generate document-level recommendations."""
        recommendations = []
        
        high_risk_count = sum(1 for r in segment_results if r.risk_category == "HIGH")
        if high_risk_count > 0:
            recommendations.append(f"Review {high_risk_count} high-risk segments immediately")
        
        citation_issues = sum(1 for r in segment_results if not r.citations_verified)
        if citation_issues > 0:
            recommendations.append(f"Verify {citation_issues} segments with citation issues")
        
        low_consistency = sum(1 for r in segment_results if r.semantic_consistency < 0.7)
        if low_consistency > len(segment_results) * 0.3:
            recommendations.append("Consider overall document coherence review")
        
        return recommendations
    
    def generate_report(self, 
                       analysis_result: DocumentAnalysisResult, 
                       output_path: str,
                       format: str = "json") -> None:
        """Generate and save analysis report."""
        
        if format.lower() == "json":
            report_data = {
                "document_id": analysis_result.document_id,
                "timestamp": analysis_result.timestamp,
                "overall_risk_score": analysis_result.overall_risk_score,
                "summary": analysis_result.summary,
                "segment_analysis": [
                    {
                        "text_preview": result.text_segment[:100] + "..." if len(result.text_segment) > 100 else result.text_segment,
                        "risk_score": result.risk_score,
                        "risk_category": result.risk_category,
                        "confidence": result.confidence,
                        "issues": result.issues,
                        "recommendations": result.recommendations,
                        "citations_verified": result.citations_verified,
                        "semantic_consistency": result.semantic_consistency
                    }
                    for result in analysis_result.segment_results
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Report generated: {output_path}")
    
    def analyze_text(self, text: str) -> HallucinationResult:
        """Quick analysis of a single text segment."""
        return self._analyze_segment(text)
    
    def batch_analyze(self, texts: List[str]) -> List[HallucinationResult]:
        """Analyze multiple text segments."""
        return [self._analyze_segment(text) for text in texts]
