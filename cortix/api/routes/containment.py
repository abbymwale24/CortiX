"""
CortiX Dashboard — Containment Actions Router
"""

import logging
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from cortix.database import get_session, ContainmentAction
from cortix.containment.executor import ContainmentExecutor

logger = logging.getLogger("cortix.api.routes.containment")

router = APIRouter(prefix="/containment", tags=["containment"])

# Request Schema for Manual Firewall override
class ContainmentRequest(BaseModel):
    src_ip: str
    action_id: int  # 0..5
    triggered_by: str = "ADMIN"
    duration_seconds: int = 60


def get_db():
    db = get_session()
    try:
        yield db
    finally:
        db.close()


@router.get("")
def get_containment_actions(limit: int = 50, db: Session = Depends(get_db)):
    """Retrieve history of blocks and rate limits."""
    actions = db.query(ContainmentAction).order_by(ContainmentAction.timestamp.desc()).limit(limit).all()
    return [a.to_dict() for a in actions]


@router.post("")
def trigger_manual_containment(req: ContainmentRequest, db: Session = Depends(get_db)):
    """Apply manual firewall block or limit override."""
    executor = ContainmentExecutor()
    success = executor.apply_action(req.action_id, req.src_ip)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to apply network firewall rule on firewall host")

    # Record action in DB
    action_names = {
        0: "ALLOW",
        1: "RATE_LIMIT",
        2: "TEMP_BLOCK",
        3: "QUARANTINE",
        4: "HARD_BLOCK",
        5: "HONEYPOT_REDIRECT",
    }
    
    action_record = ContainmentAction(
        src_ip=req.src_ip,
        action=action_names.get(req.action_id, "ALLOW"),
        triggered_by=req.triggered_by,
        duration_seconds=req.duration_seconds,
        resolved=False,
    )
    db.add(action_record)
    db.commit()
    
    return {"status": "SUCCESS", "action": action_record.to_dict()}


@router.post("/restore")
def restore_host(src_ip: str, db: Session = Depends(get_db)):
    """Restore host to benign state (ALLOW)."""
    executor = ContainmentExecutor()
    success = executor.remove_blocks(src_ip)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to lift firewall rules for host")

    # Update actions to resolved in DB
    db.query(ContainmentAction).filter(
        ContainmentAction.src_ip == src_ip,
        ContainmentAction.resolved == False
    ).update({ContainmentAction.resolved: True})
    
    db.commit()
    return {"status": "SUCCESS", "ip": src_ip, "restored": True}
