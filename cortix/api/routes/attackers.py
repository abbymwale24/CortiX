"""
CortiX Dashboard — Attacker Profiles Router
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from cortix.database import get_session, AttackerProfile

logger = logging.getLogger("cortix.api.routes.attackers")

router = APIRouter(prefix="/attackers", tags=["attackers"])


def get_db():
    db = get_session()
    try:
        yield db
    finally:
        db.close()


@router.get("")
def get_attackers(limit: int = 50, db: Session = Depends(get_db)):
    """Retrieve unique compiled attacker OSINT profiles."""
    profiles = db.query(AttackerProfile).order_by(AttackerProfile.last_seen.desc()).limit(limit).all()
    return [p.to_dict() for p in profiles]


@router.get("/{ip}")
def get_attacker_profile(ip: str, db: Session = Depends(get_db)):
    """Retrieve single attacker profile by unique IP address."""
    profile = db.query(AttackerProfile).filter(AttackerProfile.ip == ip).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Attacker profile not found")
    return profile.to_dict()
