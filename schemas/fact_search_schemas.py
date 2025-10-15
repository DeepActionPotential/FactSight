from dataclasses import dataclass, field
from typing import List, Optional



@dataclass
class FactCheckEntry:
    """Represents a single fact-check result from a verified source."""
    text: str
    claimant: Optional[str]
    claim_date: Optional[str]
    rating: Optional[str]
    publisher: Optional[str]
    url: Optional[str]


@dataclass
class FactCheckResult:
    """Structured result returned by the Fact Check Tools API."""
    verified: bool
    summary_verdict: str
    results: List[FactCheckEntry] = field(default_factory=list)

    def is_fake(self) -> bool:
        """Return True if the aggregated verdict indicates likely false information."""
        return self.summary_verdict.lower() in ["false", "likely false", "incorrect"]

    def summary(self) -> str:
        """Return a human-readable summary for quick inspection."""
        publishers = {r.publisher for r in self.results if r.publisher}
        return (
            f"✅ Verified: {self.verified}\n"
            f"🧾 Summary Verdict: {self.summary_verdict}\n"
            f"📚 Sources: {', '.join(publishers) if publishers else 'N/A'}"
        )

