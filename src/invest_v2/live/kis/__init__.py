"""KIS OpenAPI integration (paper/real).

This package keeps endpoint paths and TR-IDs configurable because KIS occasionally
revises them.

For initial wiring, we focus on:
- OAuth2 access token issuance
- Hashkey generation for POST bodies
- Domestic stock cash order (market)
- Account snapshot (cash + positions) for reconciliation hooks
"""
