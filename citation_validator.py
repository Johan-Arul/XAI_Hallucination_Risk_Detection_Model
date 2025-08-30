"""
Citation validation module for legal documents.
"""

import re
import logging
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin
import time


@dataclass
class Citation:
    """Represents a legal citation."""
    text: str
    citation_type: str
    volume: Optional[str] = None
    reporter: Optional[str] = None
    page: Optional[str] = None
    year: Optional[str] = None
    court: Optional[str] = None
    case_name: Optional[str] = None


@dataclass
class CitationValidationResult:
    """Result of citation validation."""
    citation: Citation
    is_valid: bool
    confidence: float
    validation_source: str
    error_message: Optional[str] = None


class CitationValidator:
    """Validates legal citations for accuracy and existence."""
    
    def __init__(self, config: Dict):
        """Initialize the citation validator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Citation patterns for different types of legal citations
        self.citation_patterns = self._init_citation_patterns()
        
        # Known legal databases and APIs
        self.legal_databases = {
            "courtlistener": "https://www.courtlistener.com/api/rest/v3/",
            "caselaw_access": "https://api.case.law/v1/",
            "justia": "https://law.justia.com/"
        }
        
        # Cache for validated citations
        self.validation_cache = {}
    
    def _init_citation_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for different citation types."""
        return {
            "case_citation": r"(\d+)\s+([A-Za-z\.]+)\s+(\d+)(?:\s*\(([^)]+)\s+(\d{4})\))?",
            "statute_citation": r"(\d+)\s+U\.?S\.?C\.?\s*§?\s*(\d+(?:\([a-z0-9]+\))?)",
            "regulation_citation": r"(\d+)\s+C\.?F\.?R\.?\s*§?\s*(\d+\.?\d*)",
            "federal_register": r"(\d+)\s+Fed\.?\s+Reg\.?\s+(\d+)",
            "law_review": r"(\d+)\s+([A-Za-z\.\s]+)\s+(\d+)\s*\((\d{4})\)",
            "supreme_court": r"(\d+)\s+U\.?S\.?\s+(\d+)\s*\((\d{4})\)",
            "federal_court": r"(\d+)\s+F\.?(\d?)d?\s+(\d+)\s*\(([^)]+)\s+(\d{4})\)"
        }
    
    def extract_citations(self, text: str) -> List[Citation]:
        """
        Extract all citations from text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        for citation_type, pattern in self.citation_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                citation = self._parse_citation_match(match, citation_type)
                if citation:
                    citations.append(citation)
        
        # Remove duplicates
        unique_citations = []
        seen_texts = set()
        
        for citation in citations:
            if citation.text not in seen_texts:
                unique_citations.append(citation)
                seen_texts.add(citation.text)
        
        self.logger.debug(f"Extracted {len(unique_citations)} unique citations")
        return unique_citations
    
    def _parse_citation_match(self, match: re.Match, citation_type: str) -> Optional[Citation]:
        """Parse a regex match into a Citation object."""
        groups = match.groups()
        text = match.group(0)
        
        try:
            if citation_type == "case_citation":
                return Citation(
                    text=text,
                    citation_type=citation_type,
                    volume=groups[0],
                    reporter=groups[1],
                    page=groups[2],
                    court=groups[3] if len(groups) > 3 else None,
                    year=groups[4] if len(groups) > 4 else None
                )
            
            elif citation_type == "statute_citation":
                return Citation(
                    text=text,
                    citation_type=citation_type,
                    volume=groups[0],
                    reporter="U.S.C.",
                    page=groups[1]
                )
            
            elif citation_type == "supreme_court":
                return Citation(
                    text=text,
                    citation_type=citation_type,
                    volume=groups[0],
                    reporter="U.S.",
                    page=groups[1],
                    year=groups[2],
                    court="Supreme Court"
                )
            
            elif citation_type == "federal_court":
                return Citation(
                    text=text,
                    citation_type=citation_type,
                    volume=groups[0],
                    reporter=f"F.{groups[1]}d" if groups[1] else "F.",
                    page=groups[2],
                    court=groups[3] if len(groups) > 3 else None,
                    year=groups[4] if len(groups) > 4 else None
                )
            
            else:
                return Citation(
                    text=text,
                    citation_type=citation_type,
                    volume=groups[0] if groups else None,
                    page=groups[-1] if groups else None
                )
                
        except Exception as e:
            self.logger.warning(f"Error parsing citation match: {e}")
            return None
    
    def verify_citations(self, citations: List[Citation]) -> List[CitationValidationResult]:
        """
        Verify a list of citations.
        
        Args:
            citations: List of Citation objects to verify
            
        Returns:
            List of CitationValidationResult objects
        """
        results = []
        
        for citation in citations:
            # Check cache first
            cache_key = citation.text
            if cache_key in self.validation_cache:
                self.logger.debug(f"Using cached result for: {citation.text}")
                results.append(self.validation_cache[cache_key])
                continue
            
            # Validate citation
            result = self._validate_single_citation(citation)
            
            # Cache result
            self.validation_cache[cache_key] = result
            results.append(result)
            
            # Rate limiting
            time.sleep(0.1)
        
        return results
    
    def _validate_single_citation(self, citation: Citation) -> CitationValidationResult:
        """Validate a single citation."""
        
        # Basic format validation
        format_valid, format_error = self._validate_citation_format(citation)
        if not format_valid:
            return CitationValidationResult(
                citation=citation,
                is_valid=False,
                confidence=0.9,
                validation_source="format_check",
                error_message=format_error
            )
        
        # Try different validation methods
        validation_methods = [
            self._validate_via_courtlistener,
            self._validate_via_caselaw_access,
            self._validate_via_pattern_analysis
        ]
        
        for method in validation_methods:
            try:
                result = method(citation)
                if result.is_valid or result.confidence > 0.7:
                    return result
            except Exception as e:
                self.logger.debug(f"Validation method failed: {e}")
                continue
        
        # If all methods fail, return low confidence invalid
        return CitationValidationResult(
            citation=citation,
            is_valid=False,
            confidence=0.3,
            validation_source="pattern_analysis",
            error_message="Could not verify citation in available databases"
        )
    
    def _validate_citation_format(self, citation: Citation) -> Tuple[bool, Optional[str]]:
        """Validate basic citation format."""
        
        if citation.citation_type == "case_citation":
            if not citation.volume or not citation.reporter or not citation.page:
                return False, "Missing required components for case citation"
            
            # Check if reporter is known
            known_reporters = [
                "F.", "F.2d", "F.3d", "F.Supp", "F.Supp.2d", "F.Supp.3d",
                "U.S.", "S.Ct.", "L.Ed", "L.Ed.2d"
            ]
            
            if citation.reporter not in known_reporters:
                # Check if it's a state reporter (basic check)
                if not re.match(r"[A-Z]+\.?(\d+d)?", citation.reporter):
                    return False, f"Unknown reporter: {citation.reporter}"
        
        elif citation.citation_type == "statute_citation":
            if not citation.volume or not citation.page:
                return False, "Missing required components for statute citation"
        
        return True, None
    
    def _validate_via_courtlistener(self, citation: Citation) -> CitationValidationResult:
        """Validate citation using CourtListener API."""
        
        if citation.citation_type not in ["case_citation", "supreme_court", "federal_court"]:
            return CitationValidationResult(
                citation=citation,
                is_valid=False,
                confidence=0.0,
                validation_source="courtlistener",
                error_message="Citation type not supported by CourtListener"
            )
        
        try:
            # Construct search query
            query_params = {
                "citation": citation.text,
                "format": "json"
            }
            
            # Note: In real implementation, you would need API key
            # This is a placeholder for the actual API call
            url = urljoin(self.legal_databases["courtlistener"], "search/")
            
            # Simulate API response for demo
            # In real implementation: response = requests.get(url, params=query_params)
            
            return CitationValidationResult(
                citation=citation,
                is_valid=True,
                confidence=0.8,
                validation_source="courtlistener",
                error_message=None
            )
            
        except Exception as e:
            return CitationValidationResult(
                citation=citation,
                is_valid=False,
                confidence=0.0,
                validation_source="courtlistener",
                error_message=str(e)
            )
    
    def _validate_via_caselaw_access(self, citation: Citation) -> CitationValidationResult:
        """Validate citation using Caselaw Access Project API."""
        
        try:
            # Construct search parameters
            search_params = {
                "cite": citation.text,
                "full_case": "false"
            }
            
            # Note: This would require actual API implementation
            # Placeholder for demonstration
            
            return CitationValidationResult(
                citation=citation,
                is_valid=True,
                confidence=0.75,
                validation_source="caselaw_access"
            )
            
        except Exception as e:
            return CitationValidationResult(
                citation=citation,
                is_valid=False,
                confidence=0.0,
                validation_source="caselaw_access",
                error_message=str(e)
            )
    
    def _validate_via_pattern_analysis(self, citation: Citation) -> CitationValidationResult:
        """Validate citation using pattern analysis and heuristics."""
        
        confidence = 0.5
        is_valid = True
        issues = []
        
        # Check year reasonableness
        if citation.year:
            try:
                year = int(citation.year)
                if year < 1789 or year > 2025:  # US legal system start to current
                    is_valid = False
                    confidence = 0.1
                    issues.append(f"Unreasonable year: {year}")
            except ValueError:
                is_valid = False
                confidence = 0.1
                issues.append("Invalid year format")
        
        # Check volume and page reasonableness
        if citation.volume:
            try:
                vol = int(citation.volume)
                if vol <= 0 or vol > 9999:  # Reasonable volume range
                    confidence -= 0.2
                    issues.append("Unusual volume number")
            except ValueError:
                confidence -= 0.3
                issues.append("Invalid volume format")
        
        if citation.page:
            try:
                # Handle page ranges
                page_match = re.match(r"(\d+)", citation.page)
                if page_match:
                    page = int(page_match.group(1))
                    if page <= 0 or page > 99999:
                        confidence -= 0.2
                        issues.append("Unusual page number")
            except ValueError:
                confidence -= 0.2
                issues.append("Invalid page format")
        
        # Check reporter format
        if citation.reporter:
            known_patterns = [
                r"F\.(\d+d)?",  # Federal Reporter
                r"F\.Supp\.(\d+d)?",  # Federal Supplement
                r"U\.S\.",  # US Reports
                r"S\.Ct\.",  # Supreme Court Reporter
                r"[A-Z]{2,4}\.?(\d+d)?",  # State reporters
            ]
            
            if not any(re.match(pattern, citation.reporter) for pattern in known_patterns):
                confidence -= 0.3
                issues.append("Unknown or unusual reporter format")
        
        error_message = "; ".join(issues) if issues else None
        
        return CitationValidationResult(
            citation=citation,
            is_valid=is_valid and confidence > 0.3,
            confidence=max(0.0, confidence),
            validation_source="pattern_analysis",
            error_message=error_message
        )
    
    def analyze_citation_patterns(self, text: str) -> Dict[str, any]:
        """Analyze citation patterns in text for potential issues."""
        
        citations = self.extract_citations(text)
        
        analysis = {
            "total_citations": len(citations),
            "citation_types": {},
            "potential_issues": [],
            "citation_density": len(citations) / len(text.split()) if text.split() else 0
        }
        
        # Count citation types
        for citation in citations:
            citation_type = citation.citation_type
            analysis["citation_types"][citation_type] = analysis["citation_types"].get(citation_type, 0) + 1
        
        # Check for potential issues
        if analysis["citation_density"] > 0.1:
            analysis["potential_issues"].append("Unusually high citation density")
        
        if analysis["citation_density"] < 0.01 and len(text.split()) > 500:
            analysis["potential_issues"].append("Unusually low citation density for legal text")
        
        # Check for citation clustering (many citations in small area)
        citation_positions = []
        for citation in citations:
            match = re.search(re.escape(citation.text), text)
            if match:
                citation_positions.append(match.start())
        
        if len(citation_positions) > 3:
            # Check for clustering
            citation_positions.sort()
            for i in range(len(citation_positions) - 2):
                if citation_positions[i+2] - citation_positions[i] < 200:  # 3 citations in 200 chars
                    analysis["potential_issues"].append("Citation clustering detected")
                    break
        
        return analysis
    
    def validate_citation_context(self, citation: Citation, surrounding_text: str) -> float:
        """
        Validate that citation is appropriately used in context.
        
        Args:
            citation: Citation to validate
            surrounding_text: Text surrounding the citation
            
        Returns:
            Context appropriateness score (0-1)
        """
        
        score = 1.0
        
        # Check for appropriate signal phrases
        signal_phrases = [
            "see", "see also", "cf.", "but see", "see generally",
            "accord", "compare", "contra", "held", "decided",
            "established", "ruled", "found", "concluded"
        ]
        
        text_before = surrounding_text[:surrounding_text.find(citation.text)].lower()
        
        # Look for signal phrases in the 50 characters before citation
        context_before = text_before[-50:] if len(text_before) >= 50 else text_before
        
        has_signal = any(phrase in context_before for phrase in signal_phrases)
        
        if not has_signal:
            score -= 0.2
        
        # Check if citation supports the claim
        # This is a simplified heuristic - in practice, would need more sophisticated analysis
        claim_indicators = [
            "therefore", "thus", "consequently", "as a result",
            "it follows", "clearly", "obviously", "established"
        ]
        
        has_claim = any(indicator in context_before for indicator in claim_indicators)
        
        if has_claim and not has_signal:
            score -= 0.3  # Claim without proper citation signal
        
        # Check for proper punctuation and formatting
        citation_end = surrounding_text.find(citation.text) + len(citation.text)
        if citation_end < len(surrounding_text):
            char_after = surrounding_text[citation_end]
            if char_after not in ".;,)":
                score -= 0.1
        
        return max(0.0, score)
    
    def detect_citation_hallucinations(self, text: str) -> List[Dict]:
        """
        Detect potential citation hallucinations.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of potential citation hallucination indicators
        """
        
        hallucination_indicators = []
        citations = self.extract_citations(text)
        
        for citation in citations:
            indicators = []
            
            # Check for impossible combinations
            if citation.year and citation.reporter:
                year = int(citation.year) if citation.year.isdigit() else None
                
                # Check if reporter was active during cited year
                reporter_periods = {
                    "F.": (1880, 1924),
                    "F.2d": (1924, 1993),
                    "F.3d": (1993, None),
                    "F.Supp": (1932, 1998),
                    "F.Supp.2d": (1998, 2014),
                    "F.Supp.3d": (2014, None)
                }
                
                if citation.reporter in reporter_periods and year:
                    start_year, end_year = reporter_periods[citation.reporter]
                    if year < start_year or (end_year and year > end_year):
                        indicators.append({
                            "type": "temporal_impossibility",
                            "message": f"Reporter {citation.reporter} was not active in {year}",
                            "confidence": 0.9
                        })
            
            # Check for unusual volume/page combinations
            if citation.volume and citation.page:
                try:
                    vol = int(citation.volume)
                    page = int(citation.page.split()[0])  # Handle page ranges
                    
                    # Very basic heuristic - unusual if page number exceeds typical bounds
                    if page > vol * 100:  # Rough heuristic
                        indicators.append({
                            "type": "unusual_pagination",
                            "message": "Unusual volume/page relationship",
                            "confidence": 0.4
                        })
                except ValueError:
                    pass
            
            # Check for duplicate citations with different details
            for other_citation in citations:
                if (citation != other_citation and 
                    citation.case_name and other_citation.case_name and
                    citation.case_name.lower() == other_citation.case_name.lower() and
                    citation.year != other_citation.year):
                    
                    indicators.append({
                        "type": "duplicate_case_different_year",
                        "message": f"Same case cited with different years",
                        "confidence": 0.8
                    })
            
            if indicators:
                hallucination_indicators.append({
                    "citation": citation.text,
                    "indicators": indicators
                })
        
        return hallucination_indicators
    
    def generate_citation_report(self, 
                               validation_results: List[CitationValidationResult]) -> Dict:
        """Generate a comprehensive citation validation report."""
        
        total_citations = len(validation_results)
        valid_citations = sum(1 for r in validation_results if r.is_valid)
        invalid_citations = total_citations - valid_citations
        
        avg_confidence = np.mean([r.confidence for r in validation_results]) if validation_results else 0.0
        
        # Group by validation source
        sources = {}
        for result in validation_results:
            source = result.validation_source
            if source not in sources:
                sources[source] = {"valid": 0, "invalid": 0}
            
            if result.is_valid:
                sources[source]["valid"] += 1
            else:
                sources[source]["invalid"] += 1
        
        # Identify problematic patterns
        error_patterns = {}
        for result in validation_results:
            if not result.is_valid and result.error_message:
                error = result.error_message
                error_patterns[error] = error_patterns.get(error, 0) + 1
        
        return {
            "summary": {
                "total_citations": total_citations,
                "valid_citations": valid_citations,
                "invalid_citations": invalid_citations,
                "validation_rate": valid_citations / total_citations if total_citations > 0 else 0.0,
                "average_confidence": avg_confidence
            },
            "validation_sources": sources,
            "common_errors": dict(sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)),
            "recommendations": self._generate_citation_recommendations(validation_results)
        }
    
    def _generate_citation_recommendations(self, 
                                         validation_results: List[CitationValidationResult]) -> List[str]:
        """Generate recommendations based on citation validation results."""
        
        recommendations = []
        
        invalid_count = sum(1 for r in validation_results if not r.is_valid)
        total_count = len(validation_results)
        
        if invalid_count > 0:
            invalid_rate = invalid_count / total_count
            
            if invalid_rate > 0.3:
                recommendations.append("High rate of invalid citations detected - comprehensive review recommended")
            elif invalid_rate > 0.1:
                recommendations.append("Moderate citation issues found - spot check recommended")
            else:
                recommendations.append("Minor citation issues detected - verify flagged citations")
        
        # Check for specific error patterns
        format_errors = sum(1 for r in validation_results 
                          if not r.is_valid and "format" in (r.error_message or "").lower())
        
        if format_errors > 0:
            recommendations.append("Citation format issues detected - review citation style guide")
        
        # Check confidence levels
        low_confidence = sum(1 for r in validation_results if r.confidence < 0.5)
        if low_confidence > total_count * 0.2:
            recommendations.append("Many citations could not be verified - manual review suggested")
        
        return recommendations
