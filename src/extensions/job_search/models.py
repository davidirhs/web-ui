# src/extensions/job_search/models.py
from typing import Optional
from pydantic import BaseModel, Field

class Job(BaseModel):
    """Pydantic model for representing job details."""
    title: str = Field(description="Job title.")
    link: str = Field(description="Link to the job posting.")
    company: str = Field(description="Company name.")
    fit_score: float = Field(description="Score indicating how well the job fits the user's profile (0.0 to 1.0).")
    location: Optional[str] = Field(default=None, description="Job location (optional).")
    salary: Optional[str] = Field(default=None, description="Salary information (optional).")
    fit_score_explanation: str = Field(description="Explanation of the fit score.")