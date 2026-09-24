from __future__ import annotations


TEAM_ALIASES = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich",
    "Leeds United": "Leeds",
    "Leeds United FC": "Leeds",
    "Leicester City": "Leicester",
    "Liverpool FC": "Liverpool",
    "Man Utd": "Man United",
    "Manchester City": "Man City",
    "Manchester City FC": "Man City",
    "Manchester United": "Man United",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Nottingham Forest FC": "Nott'm Forest",
    "Southampton FC": "Southampton",
    "Spurs": "Tottenham",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United": "West Ham",
    "West Ham United FC": "West Ham",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers FC": "Wolves",
}


def normalize_team_name(name: str) -> str:
    """Map provider names to the canonical Football-Data.co.uk names."""
    cleaned = " ".join(str(name).strip().split())
    return TEAM_ALIASES.get(cleaned, cleaned)
