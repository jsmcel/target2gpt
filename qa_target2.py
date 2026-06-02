#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from target2_ask import answer_question


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "output" / "qa"
REPORT_MD = REPORT_DIR / "qa_report.md"
REPORT_JSON = REPORT_DIR / "qa_report.json"

QUESTIONS = [
    {
        "id": "term-clm-definition",
        "question": "Que es CLM en TARGET2?",
        "must_contain_any": ["central liquidity management", "liquidez", "mca"],
        "min_citations": 1,
    },
    {
        "id": "term-rtgs-definition",
        "question": "Que es RTGS en T2?",
        "must_contain_any": ["real-time gross settlement", "liquidacion bruta", "pagos"],
        "min_citations": 1,
    },
    {
        "id": "term-mca-dca",
        "question": "Diferencia entre una MCA y una RTGS DCA",
        "must_contain_any": ["main cash account", "dedicated cash account", "liquidez"],
        "min_citations": 1,
    },
    {
        "id": "domain-connectivity",
        "question": "Como funciona la conectividad ESMIG en TARGET Services?",
        "must_contain_any": ["esmig", "connect", "conect"],
        "min_citations": 1,
    },
    {
        "id": "term-crdm",
        "question": "Que papel tiene CRDM en CLM y RTGS?",
        "must_contain_any": ["common reference data", "datos de referencia", "particip"],
        "min_citations": 1,
    },
    {
        "id": "message-camt-050",
        "question": "Que es un camt.050 en TARGET2?",
        "must_contain_any": ["liquiditycredittransfer", "transferencia de liquidez", "camt.050"],
        "min_citations": 1,
    },
    {
        "id": "answer-not-search-list",
        "question": "Que es RTGS?",
        "must_contain_any": [
            "real-time gross settlement",
            "liquidacion bruta",
            "dinero de banco central",
        ],
        "must_not_contain_any": ["pasajes locales", "revisaria primero"],
        "min_citations": 1,
    },
]


def run_case(case: dict) -> dict:
    payload = answer_question(case["question"], top_k=10, language="es", generate=False, model_preset="local_rag")
    text = str(payload.get("answer") or "").lower()
    citations = payload.get("citations") or []
    contains = any(term.lower() in text for term in case.get("must_contain_any", []))
    avoids_forbidden_terms = not any(term.lower() in text for term in case.get("must_not_contain_any", []))
    enough_citations = len(citations) >= int(case.get("min_citations", 0))
    ok = contains and avoids_forbidden_terms and enough_citations and payload.get("confidence") != "low"
    return {
        "id": case["id"],
        "question": case["question"],
        "ok": ok,
        "confidence": payload.get("confidence"),
        "citations": len(citations),
        "contains_expected_term": contains,
        "avoids_forbidden_terms": avoids_forbidden_terms,
        "answer_excerpt": str(payload.get("answer") or "")[:800],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run smoke QA checks against the local TARGET2 GPT RAG.")
    parser.parse_args(argv)
    results = [run_case(case) for case in QUESTIONS]
    passed = sum(1 for item in results if item["ok"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps({"passed": passed, "total": len(results), "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# TARGET2 GPT QA Report",
        "",
        f"- Passed: {passed}/{len(results)}",
        "",
    ]
    for item in results:
        mark = "PASS" if item["ok"] else "FAIL"
        lines.append(f"## {mark} {item['id']}")
        lines.append("")
        lines.append(f"- Question: {item['question']}")
        lines.append(f"- Confidence: {item['confidence']}")
        lines.append(f"- Citations: {item['citations']}")
        lines.append("")
        lines.append(item["answer_excerpt"].replace("\n", " "))
        lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"{passed}/{len(results)} passed")
    print(f"report: {REPORT_MD}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

