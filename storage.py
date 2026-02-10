# storage.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PaperState:
    paper_id: str
    filename: str


class MemoryState:
    """
    MVP uchun oddiy RAM storage.
    Productionda PostgreSQL/Redisga o‘tkazasiz.
    """
    def __init__(self) -> None:
        self.active_paper: Dict[int, PaperState] = {}  # user_id -> active paper

    def set_active(self, user_id: int, paper_id: str, filename: str) -> None:
        self.active_paper[user_id] = PaperState(paper_id=paper_id, filename=filename)

    def get_active(self, user_id: int) -> Optional[PaperState]:
        return self.active_paper.get(user_id)
