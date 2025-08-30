"""
Document processing module for handling various legal document formats.
"""

import logging
import os
from typing import Dict, List, Optional, Union
from pathlib import Path
import re

import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import pandas as pd


class DocumentProcessor:
    """Processes legal documents in various formats."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.logger = logging.getLogger(__name__)
        
        # Supported file formats
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.doc': self._process_docx,
            '.txt': self._process_txt,
            '.html': self._process_html,
            '.htm': self._process_html
        }
        
        # Legal document structure patterns
        self.structure_patterns = self._init_structure_patterns()
    
    def _init_structure_patterns(self) -> Dict[str, str]:
        """Initialize patterns for identifying document structures."""
        return {
            "contract_header": r"(AGREEMENT|CONTRACT|MEMORANDUM OF UNDERSTANDING)",
            "section_header": r"(SECTION|ARTICLE|CLAUSE)\s+([IVX\d]+)",
            "whereas_clause": r"WHEREAS,?\s+(.+?)(?=WHEREAS|NOW THEREFORE)",
            "party_definition": r"(Party|Parties?)\s+([A-Z][^.]+)",
            "signature_block": r"(IN WITNESS WHEREOF|EXECUTED|SIGNED)",
            "legal_citation": r"(\d+\s+[A-Za-z\.]+\s+\d+)",
            "statute_reference": r"(\d+\s+U\.?S\.?C\.?\s*§?\s*\d+)",
            "court_case": r"([A-Z][^v]+\s+v\.?\s+[A-Z][^,]+)",
        }
    
    def load_document(self, file_path: str) -> Dict[str, any]:
        """
        Load and process a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing processed document data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        self.logger.info(f"Processing document: {file_path}")
        
        try:
            processor_func = self.supported_formats[file_extension]
            document_data = processor_func(file_path)
            
            # Add metadata
            document_data.update({
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_extension": file_extension,
                "processing_timestamp": pd.Timestamp.now().isoformat()
            })
            
            # Analyze document structure
            structure_analysis = self._analyze_document_structure(document_data["text"])
            document_data["structure"] = structure_analysis
            
            return document_data
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def _process_pdf(self, file_path: Path) -> Dict[str, any]:
        """Process PDF document."""
        text_content = ""
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                if pdf_reader.metadata:
                    metadata = {
                        "title": pdf_reader.metadata.get('/Title', ''),
                        "author": pdf_reader.metadata.get('/Author', ''),
                        "subject": pdf_reader.metadata.get('/Subject', ''),
                        "creator": pdf_reader.metadata.get('/Creator', ''),
                        "creation_date": pdf_reader.metadata.get('/CreationDate', '')
                    }
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        self.logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                
                metadata["page_count"] = len(pdf_reader.pages)
        
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            raise
        
        return {
            "text": self._clean_text(text_content),
            "raw_text": text_content,
            "metadata": metadata,
            "document_type": "pdf"
        }
    
    def _process_docx(self, file_path: Path) -> Dict[str, any]:
        """Process DOCX document."""
        try:
            doc = Document(file_path)
            
            # Extract text from paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            text_content = "\n".join(paragraphs)
            
            # Extract metadata
            metadata = {
                "title": doc.core_properties.title or "",
                "author": doc.core_properties.author or "",
                "subject": doc.core_properties.subject or "",
                "created": doc.core_properties.created.isoformat() if doc.core_properties.created else "",
                "modified": doc.core_properties.modified.isoformat() if doc.core_properties.modified else "",
                "paragraph_count": len(paragraphs)
            }
            
            return {
                "text": self._clean_text(text_content),
                "raw_text": text_content,
                "metadata": metadata,
                "document_type": "docx",
                "paragraphs": paragraphs
            }
            
        except Exception as e:
            self.logger.error(f"Error processing DOCX: {e}")
            raise
    
    def _process_txt(self, file_path: Path) -> Dict[str, any]:
        """Process plain text document."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text_content = file.read()
            
            # Basic metadata
            metadata = {
                "line_count": len(text_content.split('\n')),
                "word_count": len(text_content.split()),
                "character_count": len(text_content)
            }
            
            return {
                "text": self._clean_text(text_content),
                "raw_text": text_content,
                "metadata": metadata,
                "document_type": "txt"
            }
            
        except Exception as e:
            self.logger.error(f"Error processing TXT: {e}")
            raise
    
    def _process_html(self, file_path: Path) -> Dict[str, any]:
        """Process HTML document."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                html_content = file.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract text content
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Extract metadata
            metadata = {
                "title": soup.title.string if soup.title else "",
                "meta_description": "",
                "meta_keywords": ""
            }
            
            # Extract meta tags
            if soup.find('meta', attrs={'name': 'description'}):
                metadata["meta_description"] = soup.find('meta', attrs={'name': 'description'}).get('content', '')
            
            if soup.find('meta', attrs={'name': 'keywords'}):
                metadata["meta_keywords"] = soup.find('meta', attrs={'name': 'keywords'}).get('content', '')
            
            return {
                "text": self._clean_text(text_content),
                "raw_text": html_content,
                "metadata": metadata,
                "document_type": "html"
            }
            
        except Exception as e:
            self.logger.error(f"Error processing HTML: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers and artifacts
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        
        # Remove extra newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _analyze_document_structure(self, text: str) -> Dict[str, any]:
        """Analyze the structure of the legal document."""
        structure = {
            "document_type": "unknown",
            "sections": [],
            "parties": [],
            "citations": [],
            "key_phrases": [],
            "structure_score": 0.0
        }
        
        # Identify document type
        structure["document_type"] = self._identify_document_type(text)
        
        # Extract sections
        structure["sections"] = self._extract_sections(text)
        
        # Extract parties
        structure["parties"] = self._extract_parties(text)
        
        # Extract citations (basic)
        structure["citations"] = self._extract_basic_citations(text)
        
        # Extract key legal phrases
        structure["key_phrases"] = self._extract_key_phrases(text)
        
        # Calculate structure score
        structure["structure_score"] = self._calculate_structure_score(structure)
        
        return structure
    
    def _identify_document_type(self, text: str) -> str:
        """Identify the type of legal document."""
        text_lower = text.lower()
        
        type_indicators = {
            "contract": ["agreement", "contract", "whereas", "party", "covenant"],
            "motion": ["motion", "court", "plaintiff", "defendant", "relief"],
            "brief": ["brief", "argument", "statement of facts", "conclusion"],
            "opinion": ["opinion", "held", "reversed", "affirmed", "remanded"],
            "statute": ["section", "subsection", "chapter", "title", "code"],
            "regulation": ["regulation", "cfr", "federal register", "rule"]
        }
        
        scores = {}
        for doc_type, indicators in type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            scores[doc_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        else:
            return "unknown"
    
    def _extract_sections(self, text: str) -> List[Dict]:
        """Extract document sections."""
        sections = []
        
        # Look for section headers
        section_pattern = self.structure_patterns["section_header"]
        matches = re.finditer(section_pattern, text, re.IGNORECASE)
        
        for match in matches:
            sections.append({
                "header": match.group(),
                "position": match.start(),
                "type": "section"
            })
        
        # Look for whereas clauses in contracts
        whereas_pattern = self.structure_patterns["whereas_clause"]
        matches = re.finditer(whereas_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            sections.append({
                "header": "WHEREAS Clause",
                "content": match.group(1)[:100] + "..." if len(match.group(1)) > 100 else match.group(1),
                "position": match.start(),
                "type": "whereas"
            })
        
        return sorted(sections, key=lambda x: x["position"])
    
    def _extract_parties(self, text: str) -> List[str]:
        """Extract party names from legal documents."""
        parties = []
        
        # Pattern for party definitions
        party_pattern = self.structure_patterns["party_definition"]
        matches = re.finditer(party_pattern, text, re.IGNORECASE)
        
        for match in matches:
            party_text = match.group(2).strip()
            if len(party_text) < 100:  # Reasonable party name length
                parties.append(party_text)
        
        # Look for case style (Plaintiff v. Defendant)
        case_pattern = self.structure_patterns["court_case"]
        matches = re.finditer(case_pattern, text)
        
        for match in matches:
            case_style = match.group().strip()
            if " v. " in case_style or " v " in case_style:
                parties.extend(case_style.split(" v. " if " v. " in case_style else " v "))
        
        # Remove duplicates and clean
        unique_parties = []
        for party in parties:
            cleaned_party = re.sub(r'[,\.].*', '', party).strip()
            if cleaned_party and cleaned_party not in unique_parties:
                unique_parties.append(cleaned_party)
        
        return unique_parties[:10]  # Limit to reasonable number
    
    def _extract_basic_citations(self, text: str) -> List[str]:
        """Extract basic citations for structure analysis."""
        citations = []
        
        citation_pattern = self.structure_patterns["legal_citation"]
        matches = re.finditer(citation_pattern, text)
        
        for match in matches:
            citations.append(match.group().strip())
        
        # Also look for statute references
        statute_pattern = self.structure_patterns["statute_reference"]
        matches = re.finditer(statute_pattern, text)
        
        for match in matches:
            citations.append(match.group().strip())
        
        return list(set(citations))  # Remove duplicates
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key legal phrases that indicate document purpose."""
        key_phrases = []
        
        # Common legal phrase patterns
        phrase_patterns = [
            r"subject to the terms",
            r"in consideration of",
            r"hereby agrees?",
            r"shall be deemed",
            r"notwithstanding",
            r"provided that",
            r"except as otherwise",
            r"to the extent",
            r"for the avoidance of doubt"
        ]
        
        for pattern in phrase_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get some context around the phrase
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                key_phrases.append(context)
        
        return key_phrases[:20]  # Limit to reasonable number
    
    def _calculate_structure_score(self, structure: Dict) -> float:
        """Calculate a score indicating how well-structured the document is."""
        score = 0.0
        
        # Points for having identifiable structure
        if structure["document_type"] != "unknown":
            score += 0.2
        
        if structure["sections"]:
            score += 0.2
            # Bonus for multiple sections
            if len(structure["sections"]) > 1:
                score += 0.1
        
        if structure["parties"]:
            score += 0.2
        
        if structure["citations"]:
            score += 0.2
            # Bonus for multiple citations
            if len(structure["citations"]) > 2:
                score += 0.1
        
        if structure["key_phrases"]:
            score += 0.1
        
        return min(1.0, score)
    
    def batch_process(self, directory_path: str) -> List[Dict]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of processed document data
        """
        directory = Path(directory_path)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")
        
        processed_documents = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    document_data = self.load_document(file_path)
                    processed_documents.append(document_data)
                    self.logger.info(f"Successfully processed: {file_path.name}")
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path.name}: {e}")
        
        self.logger.info(f"Batch processed {len(processed_documents)} documents")
        return processed_documents
    
    def extract_text_segments(self, 
                            document_data: Dict, 
                            segment_type: str = "paragraph") -> List[Dict]:
        """
        Extract text segments from processed document.
        
        Args:
            document_data: Processed document data
            segment_type: Type of segmentation ('paragraph', 'sentence', 'section')
            
        Returns:
            List of text segments with metadata
        """
        text = document_data["text"]
        segments = []
        
        if segment_type == "paragraph":
            paragraphs = text.split('\n\n')
            for i, para in enumerate(paragraphs):
                if para.strip():
                    segments.append({
                        "text": para.strip(),
                        "type": "paragraph",
                        "index": i,
                        "word_count": len(para.split()),
                        "char_count": len(para)
                    })
        
        elif segment_type == "sentence":
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for i, sent in enumerate(sentences):
                if sent.strip():
                    segments.append({
                        "text": sent.strip(),
                        "type": "sentence",
                        "index": i,
                        "word_count": len(sent.split()),
                        "char_count": len(sent)
                    })
        
        elif segment_type == "section":
            # Use structure analysis to segment by sections
            if "structure" in document_data and document_data["structure"]["sections"]:
                sections = document_data["structure"]["sections"]
                for i, section in enumerate(sections):
                    start_pos = section["position"]
                    end_pos = sections[i+1]["position"] if i+1 < len(sections) else len(text)
                    
                    section_text = text[start_pos:end_pos].strip()
                    if section_text:
                        segments.append({
                            "text": section_text,
                            "type": "section",
                            "index": i,
                            "header": section.get("header", ""),
                            "word_count": len(section_text.split()),
                            "char_count": len(section_text)
                        })
            else:
                # Fallback to paragraph segmentation
                return self.extract_text_segments(document_data, "paragraph")
        
        return segments
    
    def preprocess_for_analysis(self, document_data: Dict) -> Dict:
        """
        Preprocess document data for hallucination analysis.
        
        Args:
            document_data: Processed document data
            
        Returns:
            Preprocessed data ready for analysis
        """
        
        # Extract different segment types
        paragraphs = self.extract_text_segments(document_data, "paragraph")
        sentences = self.extract_text_segments(document_data, "sentence")
        
        # Filter segments by length (remove very short segments)
        min_word_count = 5
        
        filtered_paragraphs = [
            seg for seg in paragraphs 
            if seg["word_count"] >= min_word_count
        ]
        
        filtered_sentences = [
            seg for seg in sentences 
            if seg["word_count"] >= min_word_count
        ]
        
        # Identify potential high-risk segments
        high_risk_indicators = [
            "specifically", "exactly", "precisely", "definitively",
            "all courts agree", "never", "always", "impossible",
            "guaranteed", "certain", "undoubtedly"
        ]
        
        risk_flagged_segments = []
        for segment in filtered_paragraphs:
            text_lower = segment["text"].lower()
            risk_score = sum(1 for indicator in high_risk_indicators if indicator in text_lower)
            
            if risk_score > 0:
                segment["preliminary_risk_score"] = risk_score / len(high_risk_indicators)
                risk_flagged_segments.append(segment)
        
        return {
            "original_data": document_data,
            "paragraphs": filtered_paragraphs,
            "sentences": filtered_sentences,
            "high_risk_segments": risk_flagged_segments,
            "statistics": {
                "total_paragraphs": len(filtered_paragraphs),
                "total_sentences": len(filtered_sentences),
                "high_risk_segments": len(risk_flagged_segments),
                "avg_paragraph_length": np.mean([s["word_count"] for s in filtered_paragraphs]) if filtered_paragraphs else 0,
                "avg_sentence_length": np.mean([s["word_count"] for s in filtered_sentences]) if filtered_sentences else 0
            }
        }
    
    def detect_formatting_anomalies(self, document_data: Dict) -> List[Dict]:
        """Detect formatting anomalies that might indicate generated content."""
        
        anomalies = []
        text = document_data["text"]
        
        # Check for inconsistent spacing
        if re.search(r'\s{3,}', text):
            anomalies.append({
                "type": "inconsistent_spacing",
                "description": "Multiple consecutive spaces found",
                "severity": "low"
            })
        
        # Check for unusual punctuation patterns
        if re.search(r'[.]{2,}', text):
            anomalies.append({
                "type": "unusual_punctuation",
                "description": "Multiple consecutive periods",
                "severity": "medium"
            })
        
        # Check for mixed quote styles
        straight_quotes = len(re.findall(r'"', text))
        smart_quotes = len(re.findall(r'["""]', text))
        
        if straight_quotes > 0 and smart_quotes > 0:
            anomalies.append({
                "type": "mixed_quote_styles",
                "description": "Both straight and smart quotes present",
                "severity": "low"
            })
        
        # Check for unusual capitalization patterns
        all_caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
        if len(all_caps_words) > len(text.split()) * 0.1:  # More than 10% all caps
            anomalies.append({
                "type": "excessive_capitalization",
                "description": "Unusually high proportion of all-caps words",
                "severity": "medium"
            })
        
        return anomalies
    
    def validate_document_integrity(self, document_data: Dict) -> Dict:
        """Validate overall document integrity."""
        
        integrity_score = 1.0
        issues = []
        
        # Check text length reasonableness
        text_length = len(document_data["text"])
        if text_length < 100:
            integrity_score -= 0.3
            issues.append("Document unusually short")
        elif text_length > 1000000:  # 1MB of text
            integrity_score -= 0.1
            issues.append("Document unusually long")
        
        # Check for encoding issues
        if '�' in document_data["text"]:
            integrity_score -= 0.2
            issues.append("Text encoding issues detected")
        
        # Check structure completeness
        structure_score = document_data.get("structure", {}).get("structure_score", 0)
        if structure_score < 0.3:
            integrity_score -= 0.2
            issues.append("Poor document structure detected")
        
        # Check for formatting anomalies
        anomalies = self.detect_formatting_anomalies(document_data)
        severe_anomalies = [a for a in anomalies if a["severity"] in ["high", "medium"]]
        
        if severe_anomalies:
            integrity_score -= len(severe_anomalies) * 0.1
            issues.extend([a["description"] for a in severe_anomalies])
        
        return {
            "integrity_score": max(0.0, integrity_score),
            "issues": issues,
            "anomalies": anomalies,
            "recommendations": self._generate_integrity_recommendations(issues)
        }
    
    def _generate_integrity_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on integrity issues."""
        recommendations = []
        
        if "encoding issues" in str(issues).lower():
            recommendations.append("Re-extract document with proper encoding")
        
        if "structure" in str(issues).lower():
            recommendations.append("Verify document completeness and proper formatting")
        
        if "short" in str(issues).lower():
            recommendations.append("Verify document is complete and not truncated")
        
        if not recommendations:
            recommendations.append("Document appears to have good integrity")
        
        return recommendations
                "
