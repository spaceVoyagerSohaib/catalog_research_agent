from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

class SourceAttribution(BaseModel):
    url: Optional[str] = Field(default=None, description="Source URL")
    domain: Optional[str] = Field(default=None, description="Domain of the source")
    credibility_score: Optional[float] = Field(default=None, description="Credibility score (0-100)")
    content_snippet: Optional[str] = Field(default=None, description="Relevant excerpt")

class ComponentData(BaseModel):
    description: Optional[str]
    description_sources: List[SourceAttribution]
    active_date: Optional[str] = Field(description="Active date in ISO format (YYYY-MM-DD)")
    active_date_sources: List[SourceAttribution]
    eos_date: Optional[str] = Field(description="End of support date in ISO format (YYYY-MM-DD)")
    eos_date_sources: List[SourceAttribution]

class SearchAttempt(TypedDict):
    query: str
    mode: str
    results: dict
    confidence: float

class ResearchState(TypedDict):
    component: str
    search_history: List[SearchAttempt]
    current_results: Optional[Dict[str, Any]]
    confidence_score: float
    iteration_count: int
    verified_sources: List[str]
    failed_sources: List[str]
    termination_reason: Optional[str]
    verification_notes: Optional[str]
    output: Optional[Dict[str, Any]]

class VerificationResult(BaseModel):
    description: Optional[str]
    description_sources: List[SourceAttribution]
    active_date: Optional[str] = Field(description="Active date in ISO format (YYYY-MM-DD)")
    active_date_sources: List[SourceAttribution]
    eos_date: Optional[str] = Field(description="End of support date in ISO format (YYYY-MM-DD)")
    eos_date_sources: List[SourceAttribution]
    overall_confidence: float
    verification_notes: str 

class ActiveVerificationResult(BaseModel):
    active_date: Optional[str] = Field(description="Active/GA/Release date in ISO format (YYYY-MM-DD)")
    active_date_sources: List[SourceAttribution]
    confidence_active: float
    notes_active: str
    status_active: str = Field(description="One of: verified, ambiguous, not_found, not_applicable")

class EosVerificationResult(BaseModel):
    eos_date: Optional[str] = Field(description="End of standard support date in ISO format (YYYY-MM-DD)")
    eos_date_sources: List[SourceAttribution]
    confidence_eos: float = Field(description="Confidence for EOS determination (0-100)")
    notes_eos: str
    status_eos: str = Field(description="One of: verified, ambiguous, not_found, not_applicable")
