#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import linear_kernel


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "data" / "processed" / "index.pkl"
CODEX = shutil.which("codex.cmd") or shutil.which("codex") or "codex"
MODEL_PRESETS = {
    "codex_high": {"label": "Codex High", "reasoning": "high"},
    "codex_fast": {"label": "Codex Fast", "reasoning": "low"},
    "local_rag": {"label": "Solo RAG", "reasoning": "none"},
}
GENERATION_CONTEXT_HITS = 24
GENERATION_RETRIEVAL_HITS = 40
MAX_CONTEXT_HITS = 64
RELEASE_RE = re.compile(r"R(20\d{2})[._ -]?(NOV|OCT|JUN|MAR)", re.I)
MESSAGE_RE = re.compile(r"\b(acmt|admi|camt|pacs|reda|semt|sese|seev)\.(\d{3})(?:\.(\d{3}))?\b", re.I)
ACRONYM_RE = re.compile(r"\b[A-Z0-9]{2,12}\b")

SPANISH_LANGUAGE_HINTS = {
    "como",
    "cual",
    "cuales",
    "cuÃ¡l",
    "cuÃ¡les",
    "dame",
    "de",
    "del",
    "el",
    "en",
    "es",
    "explica",
    "hay",
    "la",
    "las",
    "los",
    "que",
    "quÃ©",
    "son",
    "una",
}
ENGLISH_LANGUAGE_HINTS = {"are", "does", "explain", "how", "in", "is", "of", "the", "what", "which", "who"}
QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "como",
    "con",
    "cual",
    "cuales",
    "cuÃ¡l",
    "cuÃ¡les",
    "de",
    "del",
    "dame",
    "el",
    "en",
    "es",
    "explica",
    "for",
    "is",
    "la",
    "las",
    "los",
    "para",
    "que",
    "quÃ©",
    "sobre",
    "the",
    "what",
    "which",
    "y",
}
FAMILY_HINTS = {
    "t2_clm_rtgs": ["t2", "clm", "rtgs", "central liquidity management", "real-time gross settlement", "mca", "rtgs dca"],
    "target_sdd": ["sdd", "scope defining", "scope", "legal basis", "service description"],
    "target_udfs": ["udfs", "functional specification", "message", "schema", "xsd", "xml"],
    "target_uhb": ["uhb", "user handbook", "gui", "screen", "u2a", "user interface"],
    "change_requests": ["change request", "cr", "crs", "release", "impact", "status"],
    "connectivity": ["connectivity", "esmig", "nsp", "swift", "message exchange", "technical"],
    "business_processes": ["business process", "bpd", "settlement", "liquidity", "payment", "lifecycle"],
    "migration_and_testing": ["migration", "testing", "readiness", "test", "certification"],
    "pricing": ["pricing", "fee", "tariff", "billing", "cost"],
    "legal": ["legal", "framework agreement", "guideline", "terms"],
    "participation": ["participant", "onboarding", "account", "party", "mca", "dca"],
    "shared_features": ["shared", "common", "crdm", "billing", "reference data"],
    "messages_and_schemas": ["iso 20022", "message", "schema", "xsd", "pacs", "camt", "admi", "reda"],
}

# TARGET2 GPT overrides: this product is focused on TARGET Services, T2,
# CLM and RTGS. The template above is intentionally overridden here.
MESSAGE_DEFINITIONS = {
    "pacs.008": {
        "name": "FIToFICustomerCreditTransfer",
        "short_es": "mensaje ISO 20022 usado para transferencias de cliente entre entidades financieras",
        "short_en": "ISO 20022 message used for customer credit transfers between financial institutions",
        "sender_es": "Lo envia una entidad participante o su agente por el canal permitido hacia T2/RTGS cuando aplica al flujo.",
        "sender_en": "It is sent by a participant or its agent through the allowed channel into T2/RTGS where applicable.",
        "purpose_es": "Sirve para iniciar o transportar una transferencia de cliente que debe procesarse en el servicio correspondiente.",
        "purpose_en": "It initiates or carries a customer credit transfer to be processed in the relevant service.",
        "flow_es": "Normalmente se valida, enruta y se confirma o rechaza mediante mensajes de estado como pacs.002.",
        "flow_en": "It is normally validated, routed and confirmed or rejected through status messages such as pacs.002.",
        "not_es": "No es un estado ni un extracto; es una instruccion de pago.",
        "not_en": "It is not a status report or statement; it is a payment instruction.",
        "related": [("pacs.002", "FIToFIPaymentStatusReport"), ("pacs.004", "PaymentReturn")],
    },
    "pacs.009": {
        "name": "FinancialInstitutionCreditTransfer",
        "short_es": "mensaje ISO 20022 para transferencias entre instituciones financieras, tipico de pagos interbancarios/RTGS",
        "short_en": "ISO 20022 message for financial institution credit transfers, typical for interbank/RTGS payments",
        "sender_es": "Lo envia un participante o agente autorizado hacia RTGS/T2 segun su configuracion.",
        "sender_en": "It is sent by an authorised participant or agent into RTGS/T2 according to its setup.",
        "purpose_es": "Sirve para ordenar un pago interbancario de alto valor o una transferencia entre instituciones financieras.",
        "purpose_en": "It orders a high-value interbank payment or a transfer between financial institutions.",
        "flow_es": "En RTGS se valida, se comprueba liquidez y se liquida en dinero de banco central si hay saldo y reglas aplicables.",
        "flow_en": "In RTGS it is validated, liquidity is checked and it settles in central bank money if balance and rules allow it.",
        "not_es": "No es gestion de liquidez; para transferencias de liquidez se usan mensajes como camt.050.",
        "not_en": "It is not liquidity management; liquidity transfers use messages such as camt.050.",
        "related": [("pacs.002", "FIToFIPaymentStatusReport"), ("camt.050", "LiquidityCreditTransfer")],
    },
    "pacs.002": {
        "name": "FIToFIPaymentStatusReport",
        "short_es": "mensaje de estado que informa aceptacion, rechazo o situacion de una instruccion de pago",
        "short_en": "status report message informing acceptance, rejection or condition of a payment instruction",
        "sender_es": "Lo envia el sistema o la entidad correspondiente como respuesta o actualizacion.",
        "sender_en": "It is sent by the system or relevant institution as a response or update.",
        "purpose_es": "Sirve para saber si un pago ha sido aceptado tecnicamente, rechazado o actualizado en su procesamiento.",
        "purpose_en": "It tells whether a payment was technically accepted, rejected or updated in processing.",
        "flow_es": "Aparece despues de mensajes como pacs.008 o pacs.009.",
        "flow_en": "It appears after messages such as pacs.008 or pacs.009.",
        "not_es": "No mueve fondos por si mismo: informa del estado.",
        "not_en": "It does not move funds by itself: it reports status.",
        "related": [("pacs.008", "FIToFICustomerCreditTransfer"), ("pacs.009", "FinancialInstitutionCreditTransfer")],
    },
    "camt.050": {
        "name": "LiquidityCreditTransfer",
        "short_es": "mensaje ISO 20022 para ordenar una transferencia de liquidez entre cuentas TARGET",
        "short_en": "ISO 20022 message used to order a liquidity transfer between TARGET accounts",
        "sender_es": "Lo envia un participante autorizado para mover liquidez entre MCA, RTGS DCA, TIPS DCA, T2S DCA u otras cuentas permitidas.",
        "sender_en": "It is sent by an authorised participant to move liquidity between MCA, RTGS DCA, TIPS DCA, T2S DCA or other allowed accounts.",
        "purpose_es": "Sirve para posicionar liquidez donde se necesita: pagos RTGS, liquidacion de sistemas vinculados, TIPS o T2S.",
        "purpose_en": "It positions liquidity where it is needed: RTGS payments, linked system settlement, TIPS or T2S.",
        "flow_es": "El sistema valida autorizaciones, cuenta origen/destino, horario y saldo antes de ejecutar o rechazar la transferencia.",
        "flow_en": "The system validates authorisation, source/destination account, timing and balance before executing or rejecting the transfer.",
        "not_es": "No es un pago de cliente: es gestion de liquidez.",
        "not_en": "It is not a customer payment: it is liquidity management.",
        "related": [("camt.025", "Receipt"), ("camt.053", "Statement"), ("camt.054", "DebitCreditNotification")],
    },
    "camt.053": {
        "name": "BankToCustomerStatement",
        "short_es": "extracto de cuenta ISO 20022 usado para reporting de movimientos y saldos",
        "short_en": "ISO 20022 account statement used for reporting entries and balances",
        "sender_es": "Lo emite el servicio correspondiente hacia el titular o actor autorizado.",
        "sender_en": "It is issued by the relevant service to the account holder or authorised actor.",
        "purpose_es": "Sirve para reconciliar saldos, movimientos y liquidez al cierre o en ventanas de reporting.",
        "purpose_en": "It supports reconciliation of balances, entries and liquidity at close or reporting windows.",
        "flow_es": "Complementa las confirmaciones y notificaciones operativas con una vision de cuenta.",
        "flow_en": "It complements operational confirmations and notifications with an account view.",
        "not_es": "No ordena pagos ni liquidez; reporta movimientos.",
        "not_en": "It does not order payments or liquidity; it reports account movements.",
        "related": [("camt.052", "AccountReport"), ("camt.054", "DebitCreditNotification")],
    },
}

TERM_DEFINITIONS = {
    "target2": {
        "triggers": ["target2", "target 2", "t2 ", "t2,", "t2.", "target services", "target"],
        "name": "TARGET2 GPT / T2",
        "short_es": "el asistente sobre TARGET Services centrado en T2, CLM y RTGS, a partir de Professional Use",
        "short_en": "the TARGET Services assistant focused on T2, CLM and RTGS, based on Professional Use",
        "role_es": "T2 es el servicio TARGET que combina CLM y RTGS para cuentas de banco central, liquidez y liquidacion bruta en tiempo real.",
        "role_en": "T2 is the TARGET service combining CLM and RTGS for central bank accounts, liquidity and real-time gross settlement.",
        "flow_es": "La respuesta debe distinguir siempre si la pregunta trata de gestion central de liquidez, pagos RTGS, conectividad o reporting.",
        "flow_en": "The answer should always distinguish whether the question is about central liquidity management, RTGS payments, connectivity or reporting.",
    },
    "clm": {
        "triggers": ["clm", "central liquidity management", "liquidity management", "gestion central de liquidez", "gestión central de liquidez"],
        "name": "CLM",
        "short_es": "Central Liquidity Management: el modulo de T2 para gestionar liquidez central y Main Cash Accounts",
        "short_en": "Central Liquidity Management: the T2 module for central liquidity and Main Cash Accounts",
        "role_es": "CLM concentra la gestion de liquidez de los participantes, incluidas MCA y transferencias hacia RTGS, TIPS o T2S cuando procede.",
        "role_en": "CLM centralises participant liquidity management, including MCAs and transfers to RTGS, TIPS or T2S where applicable.",
        "flow_es": "Operativamente, CLM abre/cierra segun calendario, recibe instrucciones de liquidez, aplica reglas y refleja saldos/reporting.",
        "flow_en": "Operationally, CLM opens/closes according to the calendar, receives liquidity instructions, applies rules and reflects balances/reporting.",
    },
    "rtgs": {
        "triggers": ["rtgs", "real-time gross settlement", "gross settlement", "liquidacion bruta", "liquidación bruta"],
        "name": "RTGS",
        "short_es": "Real-Time Gross Settlement: el componente T2 donde se liquidan pagos en dinero de banco central de forma bruta y en tiempo real",
        "short_en": "Real-Time Gross Settlement: the T2 component where payments settle in central bank money on a gross real-time basis",
        "role_es": "RTGS procesa pagos de alto valor y flujos de sistemas vinculados usando liquidez disponible en cuentas dedicadas.",
        "role_en": "RTGS processes high-value payments and linked-system flows using liquidity available on dedicated accounts.",
        "flow_es": "El flujo basico es recepcion, validacion, comprobacion de liquidez, posible cola/gestion de prioridades y liquidacion o rechazo.",
        "flow_en": "The basic flow is receipt, validation, liquidity check, possible queue/priority handling and settlement or rejection.",
    },
    "mca": {
        "triggers": ["mca", "main cash account", "main cash accounts"],
        "name": "MCA",
        "short_es": "Main Cash Account: cuenta principal de efectivo en CLM",
        "short_en": "Main Cash Account: the main cash account in CLM",
        "role_es": "Sirve como cuenta central de liquidez del participante y como punto de gobierno de saldos y transferencias.",
        "role_en": "It acts as the participant's central liquidity account and control point for balances and transfers.",
        "flow_es": "Desde una MCA se puede mover liquidez hacia cuentas dedicadas segun reglas, horarios y autorizaciones.",
        "flow_en": "Liquidity can be moved from an MCA to dedicated accounts according to rules, timing and permissions.",
    },
    "dca": {
        "triggers": ["dca", "dedicated cash account", "rtgs dca", "tips dca", "t2s dca"],
        "name": "DCA",
        "short_es": "Dedicated Cash Account: cuenta dedicada para un servicio TARGET concreto, como RTGS, TIPS o T2S",
        "short_en": "Dedicated Cash Account: an account dedicated to a specific TARGET service, such as RTGS, TIPS or T2S",
        "role_es": "Permite separar liquidez por servicio y ejecutar pagos o liquidaciones en el modulo correspondiente.",
        "role_en": "It separates liquidity by service and enables payments or settlement in the relevant module.",
        "flow_es": "La liquidez suele alimentarse desde CLM/MCA y se usa en el servicio destino segun sus reglas operativas.",
        "flow_en": "Liquidity is usually funded from CLM/MCA and used in the destination service according to operational rules.",
    },
    "esmig": {
        "triggers": ["esmig", "connectivity", "conectividad", "a2a", "u2a", "nsp"],
        "name": "ESMIG",
        "short_es": "Eurosystem Single Market Infrastructure Gateway: capa comun de conectividad para TARGET Services",
        "short_en": "Eurosystem Single Market Infrastructure Gateway: common connectivity layer for TARGET Services",
        "role_es": "Canaliza acceso A2A y U2A a servicios como T2, T2S, TIPS y ECMS mediante proveedores y configuraciones autorizadas.",
        "role_en": "It channels A2A and U2A access to services such as T2, T2S, TIPS and ECMS through authorised providers and configurations.",
        "flow_es": "En una respuesta, separa conectividad tecnica, mensajeria, autenticacion y perfil operativo del participante.",
        "flow_en": "In an answer, separate technical connectivity, messaging, authentication and participant operating profile.",
    },
    "crdm": {
        "triggers": ["crdm", "common reference data", "reference data", "datos de referencia"],
        "name": "CRDM",
        "short_es": "Common Reference Data Management: componente comun de datos de referencia de TARGET Services",
        "short_en": "Common Reference Data Management: common reference data component of TARGET Services",
        "role_es": "Mantiene datos de participantes, cuentas, autorizaciones y configuraciones que condicionan CLM/RTGS y otros servicios.",
        "role_en": "It maintains participant, account, permission and configuration data affecting CLM/RTGS and other services.",
        "flow_es": "Los cambios de datos de referencia impactan en que mensajes, cuentas y operaciones son validas.",
        "flow_en": "Reference data changes affect which messages, accounts and operations are valid.",
    },
    "bdm": {
        "triggers": ["bdm", "business day management", "business day", "dia operativo", "día operativo"],
        "name": "BDM",
        "short_es": "Business Day Management: gestion del dia operativo de TARGET Services",
        "short_en": "Business Day Management: operating day management for TARGET Services",
        "role_es": "Define fases, cambios de estado, calendarios y ventanas que condicionan pagos, liquidez y reporting.",
        "role_en": "It defines phases, status changes, calendars and windows that condition payments, liquidity and reporting.",
        "flow_es": "Para responder bien hay que ubicar la operacion en la fase del dia: apertura, cut-off, cierre o eventos excepcionales.",
        "flow_en": "A good answer places the operation in the day phase: opening, cut-off, closing or exceptional events.",
    },
    "econsii": {
        "triggers": ["econs", "econs ii", "contingency"],
        "name": "ECONS II",
        "short_es": "mecanismo de contingencia de TARGET Services para continuidad operativa en escenarios excepcionales",
        "short_en": "TARGET Services contingency mechanism for operational continuity in exceptional scenarios",
        "role_es": "Permite mantener o recuperar operativa critica bajo procedimientos de contingencia.",
        "role_en": "It supports maintaining or recovering critical operations under contingency procedures.",
        "flow_es": "Debe tratarse como flujo excepcional, no como operativa ordinaria de CLM/RTGS.",
        "flow_en": "It must be treated as an exceptional flow, not ordinary CLM/RTGS processing.",
    },
}

TERM_MATCH_PRIORITY = ["clm", "rtgs", "mca", "dca", "esmig", "crdm", "bdm", "econsii", "target2"]


def _trigger_matches_query(trigger: str, low_query: str) -> bool:
    return _trigger_match_span(trigger, low_query) is not None


def _trigger_match_span(trigger: str, low_query: str) -> tuple[int, int] | None:
    trigger_low = trigger.lower().strip()
    if not trigger_low:
        return None
    if re.fullmatch(r"[a-z0-9]{2,6}", trigger_low):
        match = re.search(rf"(?<![a-z0-9]){re.escape(trigger_low)}(?![a-z0-9])", low_query)
    else:
        match = re.search(re.escape(trigger_low), low_query)
    if not match:
        return None
    return match.start(), match.end()


def _is_definition_subject(prefix: str) -> bool:
    return re.search(r"(?:que es|qué es|what is|define|defin\w*)\s+(?:un|una|el|la|the|a|an)?\s*$", prefix) is not None


def find_term_definition(query: str) -> tuple[str, dict[str, Any]] | None:
    low = query.lower()
    priority = {key: idx for idx, key in enumerate(TERM_MATCH_PRIORITY)}
    matches: list[tuple[tuple[int, int, int, int], str, dict[str, Any]]] = []
    for candidate_key, candidate in TERM_DEFINITIONS.items():
        for trigger in candidate.get("triggers", []):
            span = _trigger_match_span(trigger, low)
            if span is None:
                continue
            start, end = span
            direct_subject = _is_definition_subject(low[:start])
            score = (
                0 if direct_subject else 1,
                start,
                -(end - start),
                priority.get(candidate_key, len(priority)),
            )
            matches.append((score, candidate_key, candidate))
    if not matches:
        return None
    _, key, definition = min(matches, key=lambda item: item[0])
    return key, definition


@dataclass
class Hit:
    rank: int
    score: float
    chunk: dict[str, Any]
    reason: str = ""

    @property
    def citation(self) -> str:
        return cite_label(self)


def detect_question_language(query: str) -> str:
    lower = query.lower()
    if re.search(r"[Â¿Â¡Ã¡Ã©Ã­Ã³ÃºÃ±]", lower):
        return "es"
    tokens = set(re.findall(r"\b[\wÃ¡Ã©Ã­Ã³ÃºÃ±]+\b", lower, flags=re.I))
    spanish_score = len(tokens & SPANISH_LANGUAGE_HINTS)
    english_score = len(tokens & ENGLISH_LANGUAGE_HINTS)
    return "en" if english_score > spanish_score else "es"


@lru_cache(maxsize=2)
def load_index(path: Path = INDEX_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Index not found at {path}. Run `python target2_ingest.py` first.")
    with path.open("rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=8)
def load_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_query(query: str) -> str:
    query = query.strip()
    replacements = {
        "liquidacion": "settlement payments liquidity RTGS CLM central bank money",
        "liquidación": "settlement payments liquidity RTGS CLM central bank money",
        "liquidaciÃ³n": "settlement payments liquidity RTGS CLM central bank money",
        "liquidez": "liquidity CLM MCA DCA liquidity transfer camt.050",
        "pago": "payment RTGS pacs.008 pacs.009 pacs.002 settlement",
        "pagos": "payments RTGS pacs.008 pacs.009 pacs.002 settlement",
        "conectividad": "connectivity ESMIG network service provider NSP",
        "pantallas": "UHB GUI screen user interface",
        "manual": "user handbook UHB",
        "requisitos": "requirements specifications SDD UDFS",
        "mensajes": "messages schemas ISO 20022",
        "mensaje": "message schema ISO 20022",
        "esquema": "schema xsd usage guideline",
        "esquemas": "schemas xsd usage guidelines",
        "campo": "field element path minOccurs maxOccurs",
        "campos": "fields elements paths minOccurs maxOccurs",
        "cr": "change request release impact status",
        "crs": "change requests release impact status",
        "clm": "central liquidity management MCA main cash account liquidity",
        "rtgs": "real-time gross settlement high value payments RTGS DCA",
        "mca": "main cash account central liquidity management CLM",
        "dca": "dedicated cash account RTGS DCA TIPS DCA T2S DCA",
        "crdm": "common reference data management participant account permissions",
        "bdm": "business day management operating day calendar cut-off",
        "econs": "ECONS II contingency TARGET Services",
    }
    tokens = re.findall(r"[\w./-]+", query, flags=re.UNICODE)
    cleaned = " ".join(token for token in tokens if token.lower() not in QUESTION_STOPWORDS)
    expanded = [cleaned or query, "TARGET Services T2 CLM RTGS MCA DCA ESMIG CRDM BDM ISO 20022 payments liquidity"]
    low = query.lower()
    for key, value in replacements.items():
        if key in low:
            expanded.append(value)
    for code in query_message_codes(query):
        expanded.append(code)
        expanded.append(code.replace(".", "_"))
    release = query_release(query)
    if release:
        expanded.append(release)
    return " ".join(expanded)


def query_release(query: str) -> str:
    match = RELEASE_RE.search(query)
    if match:
        return f"R{match.group(1)}.{match.group(2).upper()}"
    reverse = re.search(r"\b(NOV|OCT|JUN|MAR)[._ -]?(20\d{2})\b", query, re.I)
    if reverse:
        return f"R{reverse.group(2)}.{reverse.group(1).upper()}"
    return ""


def query_message_codes(query: str) -> list[str]:
    codes: list[str] = []
    for match in MESSAGE_RE.finditer(query):
        code = f"{match.group(1).lower()}.{match.group(2)}"
        if match.group(3):
            code = f"{code}.{match.group(3)}"
        if code not in codes:
            codes.append(code)
    return codes


def query_acronyms(query: str) -> list[str]:
    return [item for item in ACRONYM_RE.findall(query.upper()) if item not in {"THE", "AND", "FOR"}]


def is_domain_query(query: str) -> bool:
    low = query.lower()
    if "target2" in low or "target services" in low or re.search(r"\bt2\b", low):
        return True
    if query_release(query) or query_message_codes(query):
        return True
    domain_terms = [
        "settlement",
        "payment",
        "payments",
        "pago",
        "pagos",
        "liquidity",
        "liquidez",
        "clm",
        "rtgs",
        "mca",
        "dca",
        "crdm",
        "bdm",
        "econs",
        "uhb",
        "udfs",
        "sdd",
        "esmig",
        "main cash account",
        "dedicated cash account",
        "central liquidity management",
        "real-time gross settlement",
        "ancillary system",
        "sistema vinculado",
        "liquidacion",
        "liquidación",
        "liquidaciÃ³n",
    ]
    return any(term in low for term in domain_terms) or any(acr in {"T2", "CLM", "RTGS", "MCA", "DCA", "CRDM", "BDM", "ESMIG", "UDFS", "UHB", "SDD"} for acr in query_acronyms(query))


def _bm25_scores(index: dict[str, Any], query: str) -> np.ndarray:
    vectorizer = index.get("bm25_vectorizer")
    matrix = index.get("bm25_matrix")
    idf = index.get("bm25_idf")
    doc_len = index.get("bm25_doc_len")
    avgdl = float(index.get("bm25_avgdl") or 1.0)
    if vectorizer is None or matrix is None or idf is None or doc_len is None:
        return np.zeros(len(index.get("chunks", [])), dtype=float)
    q = vectorizer.transform([query])
    if q.nnz == 0:
        return np.zeros(matrix.shape[0], dtype=float)
    k1 = 1.5
    b = 0.75
    scores = np.zeros(matrix.shape[0], dtype=float)
    for term_idx in q.indices:
        col = matrix[:, term_idx].tocoo()
        if col.nnz == 0:
            continue
        freq = col.data.astype(float)
        denom = freq + k1 * (1 - b + b * (doc_len[col.row] / avgdl))
        scores[col.row] += idf[term_idx] * (freq * (k1 + 1) / denom)
    max_score = float(scores.max() or 0.0)
    if max_score:
        scores = scores / max_score
    return scores


def _chunk_text_for_ranking(chunk: dict[str, Any]) -> str:
    return "\n".join(
        str(chunk.get(key) or "")
        for key in [
            "title",
            "family",
            "category",
            "release",
            "unit_type",
            "unit",
            "message_id",
            "usage_guideline_name",
            "collection",
            "local_path",
            "source_url",
            "text",
        ]
    )


def metadata_bonus(chunk: dict[str, Any], query: str) -> float:
    low = query.lower()
    hay = _chunk_text_for_ranking(chunk).lower()
    bonus = 0.0
    family = str(chunk.get("family") or "")
    for hinted_family, hints in FAMILY_HINTS.items():
        if family == hinted_family and any(hint in low for hint in hints):
            bonus += 0.18
    release = query_release(query)
    if release:
        bonus += 0.35 if chunk.get("release") == release else -0.05
    for code in query_message_codes(query):
        aliases = {code, code.replace(".", "_"), code.replace(".", " ")}
        if any(alias in hay for alias in aliases):
            bonus += 0.35
    for acronym in query_acronyms(query):
        if re.search(rf"\b{re.escape(acronym.lower())}\b", hay):
            bonus += 0.08
    if chunk.get("revision_status") == "clean":
        bonus += 0.04
    return bonus


def retrieve(index: dict[str, Any], query: str, top_k: int = 8, pool: int = 180) -> list[Hit]:
    chunks = index.get("chunks") or []
    if not chunks:
        return []
    normalized = normalize_query(query)
    word_scores = linear_kernel(index["word_vectorizer"].transform([normalized]), index["word_matrix"]).ravel()
    char_scores = linear_kernel(index["char_vectorizer"].transform([normalized]), index["char_matrix"]).ravel()
    bm25_scores = _bm25_scores(index, normalized)
    scores = 0.48 * word_scores + 0.25 * char_scores + 0.27 * bm25_scores
    pool_size = min(max(pool, top_k * 6), len(chunks))
    candidate_idx = np.argpartition(scores, -pool_size)[-pool_size:]
    ranked = sorted(candidate_idx, key=lambda idx: scores[idx] + metadata_bonus(chunks[idx], query), reverse=True)
    hits: list[Hit] = []
    for rank, idx in enumerate(ranked[:top_k], start=1):
        score = float(scores[idx] + metadata_bonus(chunks[idx], query))
        hits.append(Hit(rank=rank, score=score, chunk=chunks[int(idx)], reason="hybrid"))
    return hits


def augment_hits(index: dict[str, Any], query: str, hits: list[Hit]) -> list[Hit]:
    return hits


def expand_neighbor_hits(index: dict[str, Any], hits: list[Hit], max_neighbors: int = 1, max_total: int = MAX_CONTEXT_HITS) -> list[Hit]:
    chunks = index.get("chunks") or []
    chunk_id_to_pos = index.get("chunk_id_to_pos") or {}
    selected: list[Hit] = []
    seen: set[str] = set()

    def add(hit: Hit) -> None:
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit.chunk))
        if chunk_id not in seen and len(selected) < max_total:
            selected.append(hit)
            seen.add(chunk_id)

    for hit in hits:
        add(hit)
        pos = chunk_id_to_pos.get(hit.chunk.get("chunk_id"))
        if pos is None:
            continue
        for offset in range(1, max_neighbors + 1):
            for neighbor_pos in (pos - offset, pos + offset):
                if 0 <= neighbor_pos < len(chunks):
                    neighbor = chunks[neighbor_pos]
                    if neighbor.get("doc_id") == hit.chunk.get("doc_id"):
                        add(Hit(rank=len(selected) + 1, score=max(hit.score - 0.08 * offset, 0.01), chunk=neighbor, reason="neighbor"))
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return selected


def context_pointer_priority(query: str, hit: Hit) -> tuple[float, float]:
    chunk = hit.chunk
    hay = _chunk_text_for_ranking(chunk).lower()
    priority = metadata_bonus(chunk, query)
    if hit.reason == "neighbor":
        priority -= 0.08
    query_terms = [term for term in re.findall(r"[\w./-]{3,}", query.lower()) if term not in QUESTION_STOPWORDS]
    priority += min(sum(0.025 for term in query_terms if term in hay), 0.5)
    return priority, hit.score


def rerank_context_hits(query: str, hits: list[Hit], max_total: int | None = None) -> list[Hit]:
    ranked = sorted(enumerate(hits), key=lambda item: (*context_pointer_priority(query, item[1]), -item[0]), reverse=True)
    selected: list[Hit] = []
    seen_chunks: set[str] = set()
    seen_texts: set[str] = set()
    for _, hit in ranked:
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit.chunk))
        text_key = re.sub(r"\W+", " ", str(hit.chunk.get("text") or "").lower())[:700]
        if chunk_id in seen_chunks or (text_key and text_key in seen_texts):
            continue
        seen_chunks.add(chunk_id)
        if text_key:
            seen_texts.add(text_key)
        selected.append(hit)
        if max_total and len(selected) >= max_total:
            break
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return selected


def trim_excerpt(text: str, max_chars: int = 950) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_stop = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(": "))
    if last_stop > 350:
        return cut[: last_stop + 1].strip()
    return cut.rstrip() + "..."


GENERIC_SOURCE_TITLES = {"english", "version 1.0", "version 1.1", "version 1.1.3"}


def _title_from_source_path(chunk: dict[str, Any]) -> str | None:
    source = str(chunk.get("source_url") or chunk.get("local_path") or "")
    if not source:
        return None
    name = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name = re.sub(r"__[^.]+", "", name)
    name = re.sub(r"\.(pdf|html|xlsx|zip)$", "", name, flags=re.I)
    name = re.sub(r"\.en$", "", name, flags=re.I)
    low = name.lower()
    if "clm-uhb" in low:
        return "T2 User Handbook R2026.JUN - Central Liquidity Management (CLM)"
    if "rtgs-uhb" in low:
        return "T2 User Handbook R2026.JUN - Real-Time Gross Settlement (RTGS)"
    if "econs" in low and "uhb" in low:
        return "T2 User Handbook R2026.JUN - ECONSII"
    replacements = [
        (r"[_-]?clean[_-]?\d{8}", ""),
        (r"[_-]?rev[_-]?\d{8}", ""),
        (r"[_-]?revised[_-]?\d{8}", ""),
        (r"\.pdf$", ""),
    ]
    for pattern, repl in replacements:
        name = re.sub(pattern, repl, name, flags=re.I)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        return None
    return name.upper() if len(name) <= 8 else name


def _display_title(chunk: dict[str, Any]) -> str:
    title = str(chunk.get("title") or "").strip()
    if not title or title.lower() in GENERIC_SOURCE_TITLES:
        return _title_from_source_path(chunk) or title or "Untitled"
    return title


def cite_label(hit: Hit) -> str:
    chunk = hit.chunk
    title = _display_title(chunk)
    release = chunk.get("release")
    unit_type = chunk.get("unit_type") or "unit"
    unit = chunk.get("unit")
    release_part = f", {release}" if release else ""
    where = f"{unit_type} {unit}" if unit not in (None, "") else unit_type
    return f"{title}{release_part}, {where}"


def citations_from_hits(hits: list[Hit]) -> list[dict[str, Any]]:
    citations = []
    for n, hit in enumerate(hits, start=1):
        citations.append(
            {
                "n": n,
                "title": _display_title(hit.chunk),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )
    return citations


def source_block(citations: list[dict[str, Any]], language: str) -> str:
    if not citations:
        return ""
    title = "Sources" if language == "en" else "Fuentes"
    return "\n\n" + title + ":\n" + "\n".join(f"[{c['n']}] {c['label']}" for c in citations)


def _term_citation_score(hit: Hit, key: str, definition: dict[str, Any]) -> float:
    hay = _chunk_text_for_ranking(hit.chunk).lower()
    title = _display_title(hit.chunk).lower()
    source = str(hit.chunk.get("source_url") or hit.chunk.get("local_path") or "").lower()
    family = str(hit.chunk.get("family") or "").lower()
    score = hit.score
    aliases = {key, str(definition.get("name") or "").lower()}
    aliases.update(trigger.lower() for trigger in definition.get("triggers", []) if len(trigger) > 3)
    if any(alias and alias in hay for alias in aliases):
        score += 0.45
    if any(alias and alias in title for alias in aliases):
        score += 0.75
    if family in {"target_uhb", "target_udfs", "target_urd"}:
        score += 0.35
    if key == "clm":
        if "central liquidity management" in title or "clm-uhb" in source or "clm_uhb" in source:
            score += 1.4
        if "clm udfs" in source or "clm_udfs" in source:
            score += 0.9
    elif key == "rtgs":
        if "real-time gross settlement" in title or "rtgs-uhb" in source or "rtgs_uhb" in source:
            score += 1.4
        if "rtgs udfs" in source or "rtgs_udfs" in source:
            score += 0.9
    elif key in {"mca", "dca"}:
        if "clm" in source or "central liquidity management" in title:
            score += 0.8
        if "rtgs" in source or "real-time gross settlement" in title:
            score += 0.45
    elif key in {"crdm", "bdm", "esmig", "econsii"}:
        if key.replace("ii", "") in title or key.replace("ii", "") in source:
            score += 1.0
    if key in {"clm", "rtgs", "mca", "dca"} and family in {"t2s", "tips"}:
        score -= 0.6
    return score


def _select_term_citation_hits(key: str, definition: dict[str, Any], hits: list[Hit], max_items: int = 4) -> list[Hit]:
    ranked = sorted(hits, key=lambda hit: _term_citation_score(hit, key, definition), reverse=True)
    selected: list[Hit] = []
    seen: set[tuple[str, Any]] = set()
    for hit in ranked:
        identity = (str(hit.chunk.get("doc_id") or ""), hit.chunk.get("unit"))
        if identity in seen:
            continue
        selected.append(hit)
        seen.add(identity)
        if len(selected) >= max_items:
            break
    return selected


def infer_question_intent(query: str, chat_history: list[dict[str, str]] | None = None) -> str:
    low = query.lower()
    if any(term in low for term in ["que es", "quÃ© es", "what is", "define", "defin"]):
        return "definition"
    if any(term in low for term in ["diferencia", "compara", "compare", "versus", " vs "]):
        return "comparison"
    if any(term in low for term in ["flujo", "flow", "paso", "step", "secuencia"]):
        return "flow"
    if any(term in low for term in ["impacto", "impact", "cambia", "change", "cr ", "change request"]):
        return "impact"
    if any(term in low for term in ["resumen", "summary", "sintetiza"]):
        return "summary"
    if chat_history and any(term in low for term in ["y ", "entonces", "mas", "mÃ¡s", "tambien", "tambiÃ©n", "eso", "este", "esta"]):
        return "follow_up"
    return "answer"


def compact_evidence_sentence(text: str, max_chars: int = 360) -> str:
    text = trim_excerpt(text, max_chars=max_chars)
    text = re.sub(r"^[â€¢\-\d.\s]+", "", text).strip()
    return text


def build_term_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    match = find_term_definition(query)
    if not match:
        return None
    key, definition = match
    citations = citations_from_hits(_select_term_citation_hits(key, definition, hits, max_items=min(4, len(hits))))
    if language == "en":
        answer = (
            f"{definition['name']} is {definition['short_en']}.\n\n"
            f"- **Operational role:** {definition['role_en']}\n"
            f"- **How it fits in TARGET2/T2:** {definition['flow_en']}\n"
            f"- **Practical reading:** in an answer, explain its function, the actors involved and the point of the TARGET2/T2 flow where it matters."
        )
    else:
        answer = (
            f"{definition['name']} es {definition['short_es']}.\n\n"
            f"- **Papel operativo:** {definition['role_es']}\n"
            f"- **Como encaja en TARGET2/T2:** {definition['flow_es']}\n"
            f"- **Lectura practica:** en una respuesta hay que explicar su funcion, los actores implicados y el punto del flujo TARGET2/T2 donde importa."
        )
    answer += source_block(citations, language)
    return {"answer": answer, "citations": citations, "confidence": "high", "answer_type": "term_definition"}


def build_synthetic_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any]:
    top = hits[: min(5, len(hits))]
    citations = citations_from_hits(top)
    intent = infer_question_intent(query)
    excerpts = [compact_evidence_sentence(str(hit.chunk.get("text") or "")) for hit in top]
    excerpts = [item for item in excerpts if item]
    subject = query.strip().rstrip("?") or ("the requested TARGET2/T2 topic" if language == "en" else "el tema TARGET2/T2 preguntado")

    if language == "en":
        if intent == "comparison":
            lead = f"The key comparison for `{subject}` is this:"
            sections = ["- **Difference:** " + (excerpts[0] if excerpts else "The retrieved local evidence is not explicit enough to state a clean difference."),
                        "- **Operational effect:** " + (excerpts[1] if len(excerpts) > 1 else "Use context mode to inspect the detailed evidence."),
                        "- **Bottom line:** treat the result as an operational distinction and verify implementation detail in the cited sources."]
        elif intent in {"flow", "follow_up"}:
            lead = f"The operational flow for `{subject}` is:"
            sections = [f"{i + 1}. {excerpt}" for i, excerpt in enumerate(excerpts[:4])]
            sections.append("Bottom line: the relevant answer is the processing sequence; the cited sources provide conditions and implementation detail.")
        else:
            lead = f"Short answer: `{subject}` is a TARGET2/T2 topic that must be explained by function, actors and operational effect."
            sections = ["- **What it means:** " + (excerpts[0] if excerpts else "The local index retrieved related TARGET2/T2 evidence, but not a single explicit definition."),
                        "- **Why it matters:** " + (excerpts[1] if len(excerpts) > 1 else "It affects how the relevant TARGET2/T2 process, actor, message or release is interpreted."),
                        "- **Practical conclusion:** the answer gives the operational synthesis; the sources are there to audit the detail."]
    else:
        if intent == "comparison":
            lead = f"La comparacion clave sobre `{subject}` es esta:"
            sections = ["- **Diferencia:** " + (excerpts[0] if excerpts else "La evidencia local recuperada no formula una diferencia unica y limpia."),
                        "- **Efecto operativo:** " + (excerpts[1] if len(excerpts) > 1 else "Usa el modo Contexto para revisar el detalle documental."),
                        "- **Conclusion practica:** tratala como una distincion operativa y valida el detalle de implementacion en las fuentes."]
        elif intent in {"flow", "follow_up"}:
            lead = f"El flujo operativo sobre `{subject}` es:"
            sections = [f"{i + 1}. {excerpt}" for i, excerpt in enumerate(excerpts[:4])]
            sections.append("Conclusion: la respuesta relevante es la secuencia de procesamiento; las fuentes aportan condiciones y detalle de implementacion.")
        else:
            lead = f"Respuesta corta: `{subject}` es un tema TARGET2/T2 que hay que explicar por funcion, actores y efecto operativo."
            sections = ["- **Que significa:** " + (excerpts[0] if excerpts else "El indice local recupero evidencia relacionada, pero no una definicion unica literal."),
                        "- **Por que importa:** " + (excerpts[1] if len(excerpts) > 1 else "Afecta a como se interpreta el proceso, actor, mensaje o release TARGET2/T2 relacionado."),
                        "- **Conclusion practica:** la respuesta da la sintesis operativa; las fuentes quedan para verificar el detalle."]

    answer = lead + "\n\n" + "\n".join(sections)
    answer += source_block(citations, language)
    confidence = "high" if hits[0].score >= 0.25 else "medium" if hits[0].score >= 0.12 else "low"
    return {"answer": answer, "citations": citations, "confidence": confidence, "answer_type": f"synthetic_{intent}"}


def _message_definition_key(code: str) -> str:
    parts = code.lower().split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else code.lower()


def _message_citation_score(hit: Hit, code: str, definition: dict[str, Any]) -> float:
    hay = _chunk_text_for_ranking(hit.chunk).lower()
    aliases = {code, code.replace(".", "_"), code.replace(".", " ")}
    score = hit.score
    if any(alias in hay for alias in aliases):
        score += 0.8
    name = str(definition.get("name") or "").lower()
    if name and name in hay:
        score += 0.6
    if "payment type" in hay and "settlement quantity" in hay:
        score += 0.35
    if "used to allow the instructing party" in hay or "request a transfer of securities" in hay:
        score += 0.55
    if "validation status" in hay or "matching status" in hay:
        score += 0.25
    if "allegement" in hay:
        score += 0.15
    return score


def _select_message_citation_hits(code: str, definition: dict[str, Any], hits: list[Hit], max_items: int = 4) -> list[Hit]:
    if not hits:
        return []
    ranked = sorted(hits, key=lambda hit: _message_citation_score(hit, code, definition), reverse=True)
    selected: list[Hit] = []
    seen_docs: set[tuple[str, Any]] = set()
    for hit in ranked:
        key = (str(hit.chunk.get("doc_id") or ""), hit.chunk.get("unit"))
        if key in seen_docs:
            continue
        selected.append(hit)
        seen_docs.add(key)
        if len(selected) >= max_items:
            break
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return selected


def build_message_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    codes = query_message_codes(query)
    if not codes:
        return None
    code = _message_definition_key(codes[0])
    definition = MESSAGE_DEFINITIONS.get(code)
    if not definition:
        return None

    citation_hits = _select_message_citation_hits(code, definition, hits)
    citations = citations_from_hits(citation_hits)

    if language == "en":
        related = "\n".join(f"- `{rel_code}`: {rel_name}" for rel_code, rel_name in definition.get("related", []))
        answer = (
            f"`{code}` is the ISO 20022 `{definition['name']}` message: {definition['short_en']}.\n\n"
            f"In TARGET2/T2 terms:\n"
            f"- **Operational role:** {definition['purpose_en']}\n"
            f"- **Sender/receiver:** {definition['sender_en']}\n"
            f"- **Lifecycle:** {definition['flow_en']}\n"
            f"- **Do not confuse it with:** {definition['not_en']}\n\n"
            f"Related messages:\n{related}"
        )
    else:
        related = "\n".join(f"- `{rel_code}`: {rel_name}" for rel_code, rel_name in definition.get("related", []))
        answer = (
            f"`{code}` es el mensaje ISO 20022 `{definition['name']}`: {definition['short_es']}.\n\n"
            f"En TARGET2/T2, en concreto:\n"
            f"- **Papel operativo:** {definition['purpose_es']}\n"
            f"- **Quien lo envia:** {definition['sender_es']}\n"
            f"- **Flujo:** {definition['flow_es']}\n"
            f"- **No lo confundas con:** {definition['not_es']}\n\n"
            f"Mensajes relacionados:\n{related}"
        )

    answer += source_block(citations, language)
    return {"answer": answer, "citations": citations, "confidence": "high", "answer_type": "message_definition"}


def build_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any]:
    if not is_domain_query(query):
        msg = (
            "I cannot see a TARGET2, T2, CLM, RTGS or TARGET Services term, message, acronym, release or process in that question. Ask with the specific concept and I will answer from the local corpus."
            if language == "en"
            else "No veo un termino, mensaje, acronimo, release o proceso TARGET2/T2/CLM/RTGS en esa pregunta. Pon el concepto concreto y respondo con el corpus local."
        )
        return {"answer": msg, "citations": [], "confidence": "low", "skip_generation": True}
    if not hits:
        msg = "I cannot find enough evidence in the local TARGET2 GPT index." if language == "en" else "No aparece evidencia suficiente en el indice local de TARGET2 GPT."
        return {"answer": msg, "citations": [], "confidence": "low"}
    message_answer = build_message_answer(query, hits, language=language)
    if message_answer:
        return message_answer
    term_answer = build_term_answer(query, hits, language=language)
    if term_answer:
        return term_answer
    return build_synthetic_answer(query, hits, language=language)


def prioritize_generation_hits(query: str, hits: list[Hit], max_hits: int) -> list[Hit]:
    reranked = rerank_context_hits(query, hits, max_total=max(len(hits), max_hits))
    return reranked[:max_hits]


def build_codex_context(query: str, hits: list[Hit], max_hits: int = GENERATION_CONTEXT_HITS) -> dict[str, Any]:
    hits = prioritize_generation_hits(query, hits, max_hits=max_hits)
    evidence = []
    for n, hit in enumerate(hits[:max_hits], start=1):
        evidence.append(
            {
                "ref": n,
                "score": round(hit.score, 4),
                "retrieval_reason": hit.reason or "hybrid",
                "citation": cite_label(hit),
                "title": hit.chunk.get("title"),
                "family": hit.chunk.get("family"),
                "release": hit.chunk.get("release"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "excerpt": trim_excerpt(hit.chunk.get("text", ""), max_chars=3000),
            }
        )
    return {
        "question": query,
        "retrieval_pipeline": "hybrid TF-IDF word + TF-IDF char + BM25 + metadata boosts + local rerank + neighbor expansion",
        "instructions_for_codex_high": (
            "Answer using only the local TARGET2/T2/CLM/RTGS evidence dossier. Cite every substantive claim with [n]. "
            "If evidence is weak or absent, say so."
        ),
        "evidence": evidence,
    }


def format_chat_history(chat_history: list[dict[str, str]] | None, language: str) -> str:
    if not chat_history:
        return ""
    labels = {"user": "Usuario" if language == "es" else "User", "assistant": "Asistente" if language == "es" else "Assistant"}
    lines: list[str] = []
    for turn in chat_history[-10:]:
        role = str(turn.get("role", "")).lower()
        if role not in labels:
            continue
        content = re.sub(r"\s+", " ", str(turn.get("content", ""))).strip()
        if content:
            lines.append(f"{labels[role]}: {content[:1600]}")
    if not lines:
        return ""
    title = (
        "Conversacion reciente, solo para resolver referencias e intencion de seguimiento:"
        if language == "es"
        else "Recent conversation, only to resolve follow-up references and intent:"
    )
    return title + "\n" + "\n".join(lines)


def build_generation_prompt(
    query: str,
    hits: list[Hit],
    language: str = "es",
    max_hits: int = GENERATION_CONTEXT_HITS,
    chat_history: list[dict[str, str]] | None = None,
    draft_answer: str | None = None,
    context_query: str | None = None,
) -> str:
    if language == "auto":
        language = detect_question_language(query)
    lang_name = "Spanish" if language == "es" else "English"
    ranking_query = context_query or query
    context = build_codex_context(ranking_query, hits, max_hits=max_hits)
    evidence_lines = []
    for item in context["evidence"]:
        evidence_lines.append(
            "\n".join(
                [
                    f"[{item['ref']}] {item['citation']}",
                    f"Retrieval: {item['retrieval_reason']} | score={item['score']}",
                    f"Local path: {item['local_path']}",
                    f"Source URL: {item['source_url']}",
                    f"Excerpt: {item['excerpt']}",
                ]
            )
        )
    history_block = format_chat_history(chat_history, language)
    intent = infer_question_intent(query, chat_history)
    draft_block = ""
    if draft_answer:
        draft_block = (
            "Structured local draft. Use it as a safety rail for terminology and coverage. "
            "Improve it if it is too short, too list-like or too shallow; do not copy it mechanically:\n"
            + re.sub(r"\s+", " ", draft_answer).strip()[:5000]
            + "\n\n"
        )
    style_rules = (
        "Write in Spanish with a natural technical-assistant voice. Start with the answer, then develop it. "
        "Analyse the user's intention before writing: if the user asks 'que es', define; if they ask for impact, explain impact; if they ask for a flow, sequence the process; if it is a follow-up, use the conversation to resolve what 'eso', 'este' or 'lo anterior' refers to. "
        "Do not answer as a document search result. The sources support the answer; they are not the answer. "
        "Keep the main answer concise and concrete unless the user asks for detail. Put sources only at the end, not inline inside the explanatory paragraphs. "
        "Use official TARGET Services, T2, CLM and RTGS names in English when the documents use them. If something is missing, say 'No aparece en la documentacion local recuperada'."
        if language == "es"
        else "Write in English with a natural, direct technical-assistant voice. Start with the answer, then develop it. "
        "Analyse the user's intention before writing: if they ask what something is, define it; if they ask for impact, explain impact; if they ask for a flow, sequence the process; if it is a follow-up, use the conversation to resolve references. "
        "Do not answer as a document search result. The sources support the answer; they are not the answer. "
        "Keep the main answer concise and concrete unless the user asks for detail. Put sources only at the end, not inline inside explanatory paragraphs. "
        "Use official TARGET Services, T2, CLM and RTGS names. If something is missing, say 'I cannot find it in the retrieved local documentation'."
    )
    return f"""You are a senior TARGET2 GPT documentation assistant running inside a local TARGET Professional Use documentation repository.

Answer the user's question in {lang_name}, naturally and directly, like a high-quality ChatGPT answer.
Use only the local TARGET Professional Use corpus: the evidence dossier below and the listed local paths as read-only pointers. Do not invent facts and do not use the internet.
Do not dump raw excerpts. Synthesize the answer.
The retrieval layer is deliberately generous. Treat it as a set of pointers to the right documents, then use your own reasoning to connect the facts, resolve follow-up references and produce the best answer.
Intent detected by the product: {intent}
Put references at the end with title, page/unit, and local path. Do not make sources the answer.
{style_rules}

User question:
{query}

{history_block + chr(10) if history_block else ""}Local evidence dossier:

{draft_block}
{chr(10).join(evidence_lines)}
"""


def generate_with_codex(
    query: str,
    hits: list[Hit],
    language: str = "es",
    timeout: int | None = None,
    chat_history: list[dict[str, str]] | None = None,
    model_preset: str = "codex_high",
    draft_answer: str | None = None,
    context_query: str | None = None,
) -> str:
    if os.environ.get("TARGET2_DISABLE_CODEX", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("Codex generation disabled by TARGET2_DISABLE_CODEX")
    if not shutil.which("codex.cmd") and not shutil.which("codex"):
        raise RuntimeError("codex CLI not found")
    prompt = build_generation_prompt(
        query,
        hits,
        language=language,
        max_hits=GENERATION_CONTEXT_HITS,
        chat_history=chat_history,
        draft_answer=draft_answer,
        context_query=context_query,
    )
    timeout = timeout or int(os.environ.get("TARGET2_CODEX_TIMEOUT", "180"))
    preset = MODEL_PRESETS.get(model_preset, MODEL_PRESETS["codex_high"])
    reasoning = preset["reasoning"]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        output_path = Path(tmp.name)
    cmd = [
        CODEX,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "-s",
        "read-only",
        "--color",
        "never",
        "-c",
        f'model_reasoning_effort="{reasoning}"',
        "-o",
        str(output_path),
        "-",
    ]
    model = os.environ.get("TARGET2_CODEX_MODEL", "").strip()
    if model:
        cmd[2:2] = ["-m", model]
    try:
        result = subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout, cwd=str(ROOT), check=False)
        if result.returncode != 0:
            output = (result.stderr or result.stdout or "").strip()
            if len(output) > 1800:
                output = output[:700] + "\n...\n" + output[-1000:]
            raise RuntimeError(output)
        answer = output_path.read_text(encoding="utf-8", errors="replace").strip()
        if not answer:
            raise RuntimeError("codex returned an empty answer")
        return answer
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


def _retrieve_context(query: str, top_k: int, generate: bool) -> list[Hit]:
    index = load_index()
    retrieval_k = max(top_k, GENERATION_RETRIEVAL_HITS) if generate else top_k
    hits = retrieve(index, query, top_k=retrieval_k)
    hits = augment_hits(index, query, hits)
    max_context = max(retrieval_k + 16, top_k, GENERATION_CONTEXT_HITS if generate else top_k)
    hits = expand_neighbor_hits(index, hits, max_neighbors=1, max_total=min(max_context, MAX_CONTEXT_HITS))
    return rerank_context_hits(query, hits, max_total=min(max_context, MAX_CONTEXT_HITS))


def answer_question(
    query: str,
    top_k: int = 16,
    language: str = "auto",
    generate: bool = False,
    retrieval_query: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    model_preset: str = "codex_high",
) -> dict[str, Any]:
    resolved_language = detect_question_language(query) if language == "auto" else language
    search_query = retrieval_query or query
    if model_preset == "local_rag":
        generate = False
    hits = _retrieve_context(search_query, top_k=top_k, generate=generate)
    payload = build_answer(query, hits, language=resolved_language)
    if generate and hits and payload.get("confidence") != "low":
        draft = payload.get("answer") or None
        generation_hits = prioritize_generation_hits(search_query, hits, max_hits=GENERATION_CONTEXT_HITS)
        try:
            payload["answer"] = generate_with_codex(
                query,
                generation_hits,
                language=resolved_language,
                chat_history=chat_history,
                model_preset=model_preset,
                draft_answer=draft,
                context_query=search_query,
            )
            payload["citations"] = citations_from_hits(generation_hits[: min(8, len(generation_hits))])
            payload["generated_by"] = MODEL_PRESETS.get(model_preset, MODEL_PRESETS["codex_high"])["label"].lower().replace(" ", "_")
            payload.pop("skip_generation", None)
        except Exception as exc:
            payload["generated_by"] = "fallback_extractivo"
            payload["generator_error"] = str(exc)
    elif payload.get("skip_generation"):
        payload["generated_by"] = "structured"
    elif model_preset == "local_rag":
        payload["generated_by"] = "local_rag"
    payload["question"] = query
    payload["language"] = resolved_language
    payload["model"] = model_preset
    payload["hits"] = [
        {
            "rank": hit.rank,
            "score": round(hit.score, 4),
            "reason": hit.reason,
            "citation": cite_label(hit),
            "chunk": {
                key: hit.chunk.get(key)
                for key in [
                    "chunk_id",
                    "doc_id",
                    "title",
                    "category",
                    "family",
                    "release",
                    "revision_status",
                    "unit_type",
                    "unit",
                    "local_path",
                    "source_url",
                    "context_path",
                ]
            },
            "excerpt": trim_excerpt(hit.chunk.get("text", ""), max_chars=900),
        }
        for hit in hits
    ]
    return payload


def read_question(args: argparse.Namespace) -> str:
    if args.question:
        return " ".join(args.question).strip()
    return sys.stdin.read().strip()


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Ask questions against the local TARGET2 GPT index.")
    parser.add_argument("question", nargs="*", help="question; if omitted, stdin is used")
    parser.add_argument("--json", action="store_true", help="print full JSON answer")
    parser.add_argument("--context", action="store_true", help="print optimized evidence JSON for Codex High")
    parser.add_argument("--generate", action="store_true", help="generate a conversational answer with Codex High")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lang", choices=["auto", "es", "en"], default="auto")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="codex_high", help="generation preset")
    args = parser.parse_args(argv)
    query = read_question(args)
    if not query:
        print("ERROR: empty question", file=sys.stderr)
        return 2
    try:
        hits = _retrieve_context(query, top_k=args.top_k, generate=args.generate)
        if args.context:
            print(json.dumps(build_codex_context(query, hits, max_hits=args.top_k), ensure_ascii=False, indent=2))
            return 0
        payload = answer_question(query, top_k=args.top_k, language=args.lang, generate=args.generate, model_preset=args.model)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["answer"])
        return 0
    except Exception as exc:
        language = detect_question_language(query)
        message = (
            f"I could not complete this query, but the CLI stayed alive. Try a more specific TARGET2/T2/CLM/RTGS term or run with --context. Error: {exc}"
            if language == "en"
            else f"No he podido completar esta consulta, pero el CLI sigue vivo. Prueba con un termino TARGET2/T2/CLM/RTGS mas concreto o ejecuta con --context. Error: {exc}"
        )
        if args.json:
            print(json.dumps({"question": query, "answer": message, "error": str(exc), "confidence": "low"}, ensure_ascii=False, indent=2))
            return 0
        print(message)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


