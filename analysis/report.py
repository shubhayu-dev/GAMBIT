"""
Formats a capability profile (from profile.py) into the ASCII-style report
shown in docs/20/31 as an example. Text-only; the Week 8 dashboard renders
the same numbers as HTML/charts.
"""

from typing import Dict


def format_profile_report(agent_name: str, profile: Dict) -> str:
    def pct(dim):
        v = profile[dim]["decision_accuracy_pct"]
        return f"{v:>5.1f}%" if v is not None else "  n/a "

    lines = [
        "─" * 44,
        " GAMBIT REPORT",
        "─" * 44,
        f" Agent: {agent_name}",
        f" Overall decision accuracy: {pct('overall')}  (n={profile['overall']['n']})",
        "─" * 44,
        f" Opening        {pct('opening')}   (n={profile['opening']['n']})",
        f" Middlegame     {pct('middlegame')}   (n={profile['middlegame']['n']})",
        f" Endgame        {pct('endgame')}   (n={profile['endgame']['n']})",
        f" Tactical*      {pct('tactical_proxy')}   (n={profile['tactical_proxy']['n']})",
        f" Positional*    {pct('positional_proxy')}   (n={profile['positional_proxy']['n']})",
        f" Defensive*     {pct('defensive_proxy')}   (n={profile['defensive_proxy']['n']})",
        "─" * 44,
        f" Consistency (regret stdev): {profile['consistency']['regret_stdev']}",
        f" Efficiency (mean time/decision): {profile['efficiency']['mean_time_used_s']}s",
        "─" * 44,
        " * proxy dimension, not from a curated benchmark - see analysis/profile.py",
        "─" * 44,
    ]
    return "\n".join(lines)
