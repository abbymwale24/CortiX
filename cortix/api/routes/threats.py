"""
CortiX Dashboard — Threats API Router
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from cortix.database import get_session, Threat, AttackerProfile

logger = logging.getLogger("cortix.api.routes.threats")

router = APIRouter(prefix="/threats", tags=["threats"])


def get_db():
    db = get_session()
    try:
        yield db
    finally:
        db.close()


@router.get("")
def get_threats(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    attack_class: str = Query(None),
    db: Session = Depends(get_db),
):
    """Retrieve paginated, filtered threat entries."""
    query = db.query(Threat)
    if attack_class:
        query = query.filter(Threat.attack_class == attack_class)
    
    threats = query.order_by(Threat.timestamp.desc()).offset(skip).limit(limit).all()
    return [t.to_dict() for t in threats]


@router.get("/{threat_id}")
def get_threat_detail(threat_id: int, db: Session = Depends(get_db)):
    """Retrieve single threat item by unique ID including attacker profile."""
    threat = db.query(Threat).filter(Threat.id == threat_id).first()
    if not threat:
        raise HTTPException(status_code=404, detail="Threat event not found")
        
    res = threat.to_dict()
    if threat.attacker_profile:
        res["attacker_profile"] = threat.attacker_profile.to_dict()
    else:
        res["attacker_profile"] = None
        
    return res


@router.post("/{threat_id}/resolve")
def resolve_threat(threat_id: int, db: Session = Depends(get_db)):
    """Mark a threat action resolved/acknowledged."""
    threat = db.query(Threat).filter(Threat.id == threat_id).first()
    if not threat:
        raise HTTPException(status_code=404, detail="Threat event not found")
        
    threat.resolved = True
    db.commit()
    return {"status": "SUCCESS", "id": threat_id, "resolved": True}
