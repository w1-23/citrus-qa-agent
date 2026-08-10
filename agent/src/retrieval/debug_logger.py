import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

class DebugFilterLogger:
    def __init__(self):
        self._log_dir = PROJECT_ROOT / "logs" / "debug_filter"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_filter_process(self, query: str, reranked: List[Dict], threshold: float, passed: List[Dict], filtered: List[Dict]):
        now = datetime.now()
        log_path = self._log_dir / f"filter_debug_{now.strftime('%Y-%m-%d')}.log"
        
        lines = []
        lines.append(f"[{now.isoformat()}] QUERY: {query}")
        lines.append(f"Dynamic Threshold: {threshold:.4f}")
        lines.append(f"Total Reranked: {len(reranked)} | ✅ Passed: {len(passed)} | ❌ Filtered: {len(filtered)}")
        lines.append("="*80)
        
        lines.append("✅ PASSED CHUNKS (Retained):")
        for i, c in enumerate(passed, 1):
            lines.append(f"  [{i}] CE={c.get('rerank_score', 0):.4f} | {c.get('paper_id', 'N/A')} | {c.get('section_name', 'N/A')}")
            lines.append(f"      Text: {c.get('text', '')[:150].replace(chr(10), ' ')}...")
            
        lines.append("\n❌ FILTERED CHUNKS (Discarded by Dynamic Threshold):")
        if not filtered:
            lines.append("  (None)")
        for i, c in enumerate(filtered, 1):
            lines.append(f"  [{i}] CE={c.get('rerank_score', 0):.4f} | {c.get('paper_id', 'N/A')} | {c.get('section_name', 'N/A')}")
            lines.append(f"      Text: {c.get('text', '')[:150].replace(chr(10), ' ')}...")
            
        lines.append("\n" + "="*80 + "\n")
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

_debug_logger = DebugFilterLogger()

def log_filter_process(query: str, reranked: List[Dict], threshold: float, passed: List[Dict], filtered: List[Dict]):
    """Module-level convenience function."""
    _debug_logger.log_filter_process(query, reranked, threshold, passed, filtered)
