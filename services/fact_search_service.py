import requests

from schemas.fact_search_schemas import FactCheckEntry, FactCheckResult

class FactCheckService:
    """
    A service wrapper around the Google Fact Check Tools API (v1alpha1).

    Provides an interface to verify claims and obtain structured verdicts from
    fact-checking organizations (BBC, PolitiFact, FactCheck.org, etc.).

    Example:
        >>> service = FactCheckService(api_key="YOUR_API_KEY")
        >>> result = service.verify_claim("COVID-19 vaccines cause infertility")
        >>> print(result.summary())
    """

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self, api_key: str, timeout: int = 10):
        """
        Initialize the FactCheckService.

        Args:
            api_key (str): Google Fact Check Tools API key.
            timeout (int): Optional. Request timeout in seconds.
        """
        self.api_key = api_key
        self.timeout = timeout

    def verify_claim(self, claim_text: str) -> FactCheckResult:
        """
        Verify a text claim using Google's Fact Check Tools API.

        Args:
            claim_text (str): The claim or post text to fact-check.

        Returns:
            FactCheckResult: Structured response with verification details.
        """
        params = {"query": claim_text, "key": self.api_key}

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            return FactCheckResult(verified=False, summary_verdict=f"Request failed: {e}")

        if "claims" not in data or not data["claims"]:
            return FactCheckResult(verified=False, summary_verdict="Unverified")

        entries = []
        for claim in data["claims"]:
            reviews = claim.get("claimReview", [])
            for review in reviews:
                entries.append(
                    FactCheckEntry(
                        text=claim.get("text", ""),
                        claimant=claim.get("claimant"),
                        claim_date=claim.get("claimDate"),
                        rating=review.get("textualRating"),
                        publisher=review.get("publisher", {}).get("name"),
                        url=review.get("url"),
                    )
                )

        # Aggregate verdict
        verdict_texts = [e.rating.lower() for e in entries if e.rating]
        if any(v in verdict_texts for v in ["false", "incorrect", "pants on fire", "mostly false"]):
            summary_verdict = "Likely False"
        elif any(v in verdict_texts for v in ["true", "mostly true", "accurate"]):
            summary_verdict = "Likely True"
        else:
            summary_verdict = "Mixed / Unclear"

        return FactCheckResult(verified=True, summary_verdict=summary_verdict, results=entries)
