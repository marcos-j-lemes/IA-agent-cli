"""
Agent Orchestrator
──────────────────
Full pipeline: raw user input → intent → DB lookup → validation → decision.

This module is the "brain" that the agent script calls.
It does NOT execute commands — it returns a decision + full audit trail.
Execution is the caller's responsibility.

Usage:
    from agent import Agent
    agent = Agent(db_path="commands.json")
    result = agent.process("listar diretorios")
    if result["decision"] == "EXECUTE":
        os.system(result["command"])
"""

from __future__ import annotations
import json
import re
import math
from pathlib import Path
from dataclasses import dataclass

from classifier import build_classifier, IntentClassifier
from validator  import validate


# ─────────────────────────────────────────────
# COMMAND DATABASE  (your vector DB substitute)
# ─────────────────────────────────────────────
# In production this would be a real vector search (pgvector, Chroma, etc.)
# Here we use a flat JSON file + cosine similarity on TF-IDF vectors.

BUILTIN_COMMANDS = [
    # docker
    {"comando": "docker images",           "categoria": "docker",     "descricao": "Lista imagens Docker locais",              "intent": "SEARCH", "rank": 1},
    {"comando": "docker ps",               "categoria": "docker",     "descricao": "Lista containers Docker em execucao",      "intent": "SEARCH", "rank": 1},
    {"comando": "docker ps -a",            "categoria": "docker",     "descricao": "Lista todos os containers Docker",         "intent": "SEARCH", "rank": 1},
    {"comando": "docker logs <container>", "categoria": "docker",     "descricao": "Exibe logs de um container",               "intent": "QUESTION", "rank": 2},
    # filesystem
    {"comando": "ls -la",                  "categoria": "filesystem", "descricao": "Lista arquivos e diretorios detalhado",    "intent": "SEARCH", "rank": 1},
    {"comando": "ls",                      "categoria": "filesystem", "descricao": "Lista arquivos no diretorio atual",        "intent": "SEARCH", "rank": 1},
    {"comando": "find . -name '*.py'",     "categoria": "filesystem", "descricao": "Busca arquivos Python recursivamente",     "intent": "SEARCH", "rank": 2},
    {"comando": "du -sh *",                "categoria": "filesystem", "descricao": "Tamanho de cada item no diretorio",        "intent": "QUESTION", "rank": 1},
    {"comando": "mkdir <nome>",            "categoria": "filesystem", "descricao": "Cria novo diretorio",                     "intent": "CREATE", "rank": 1},
    {"comando": "touch <arquivo>",         "categoria": "filesystem", "descricao": "Cria arquivo vazio",                      "intent": "CREATE", "rank": 1},
    # process / system
    {"comando": "ps aux",                  "categoria": "process",    "descricao": "Lista todos os processos em execucao",     "intent": "SEARCH", "rank": 1},
    {"comando": "top",                     "categoria": "process",    "descricao": "Monitor de processos em tempo real",       "intent": "QUESTION", "rank": 1},
    {"comando": "df -h",                   "categoria": "system",     "descricao": "Espaco em disco disponivel",               "intent": "QUESTION", "rank": 1},
    {"comando": "free -h",                 "categoria": "system",     "descricao": "Memoria RAM disponivel",                   "intent": "QUESTION", "rank": 1},
    {"comando": "uname -a",                "categoria": "system",     "descricao": "Informacoes do sistema operacional",       "intent": "QUESTION", "rank": 1},
    {"comando": "whoami",                  "categoria": "system",     "descricao": "Usuario atual logado",                    "intent": "QUESTION", "rank": 1},
    # network
    {"comando": "ifconfig",                "categoria": "network",    "descricao": "Interfaces de rede e enderecos IP",        "intent": "QUESTION", "rank": 1},
    {"comando": "netstat -tulnp",          "categoria": "network",    "descricao": "Portas abertas e servicos em escuta",      "intent": "SEARCH", "rank": 2},
    {"comando": "ping <host>",             "categoria": "network",    "descricao": "Testa conectividade com host",             "intent": "QUESTION", "rank": 1},
    # git
    {"comando": "git status",             "categoria": "git",        "descricao": "Estado atual do repositorio Git",          "intent": "SEARCH", "rank": 1},
    {"comando": "git log --oneline",      "categoria": "git",        "descricao": "Historico de commits resumido",            "intent": "SEARCH", "rank": 1},
    {"comando": "git branch",             "categoria": "git",        "descricao": "Lista branches locais",                    "intent": "SEARCH", "rank": 1},
]


# ─────────────────────────────────────────────
# TINY VECTOR SEARCH (TF-IDF cosine)
# ─────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    words = t.split()
    return words + [words[i] + " " + words[i+1] for i in range(len(words) - 1)]


def _build_index(docs: list[str]) -> tuple[dict, dict, list[list[float]]]:
    vocab: dict[str, int] = {}
    df:    dict[str, int] = {}
    token_lists = []
    for doc in docs:
        toks = _tokenize(doc)
        token_lists.append(toks)
        seen: set[str] = set()
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)
    N = len(docs)
    idf = {t: math.log((1 + N) / (1 + f)) + 1 for t, f in df.items()}
    vectors = []
    for toks in token_lists:
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec = [0.0] * len(vocab)
        for t, cnt in tf.items():
            if t in vocab:
                vec[vocab[t]] = (1 + math.log(cnt)) * idf.get(t, 1)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vocab, idf, vectors


def _query_vector(text: str, vocab: dict, idf: dict, size: int) -> list[float]:
    toks = _tokenize(text)
    tf: dict[str, int] = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1
    vec = [0.0] * size
    for t, cnt in tf.items():
        if t in vocab:
            vec[vocab[t]] = (1 + math.log(cnt)) * idf.get(t, 1)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return round(1.0 - dot, 6)   # distance = 1 - similarity


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────

class Agent:
    """
    Stateless agent that processes a single user input through the full pipeline.
    """

    def __init__(
        self,
        db_path:         str | None = None,
        classifier_path: str | None = None,
        dataset_path:    str | None = None,
        verbose:         bool = False,
    ):
        self.verbose = verbose

        # Load command DB
        commands = list(BUILTIN_COMMANDS)
        if db_path:
            with open(db_path) as f:
                commands.extend(json.load(f))
        self._commands = commands

        # Build search index on concatenated text fields
        index_docs = [
            f"{c['descricao']} {c['categoria']} {c['comando']}"
            for c in commands
        ]
        self._vocab, self._idf, self._vecs = _build_index(index_docs)
        self._index_docs = index_docs

        # Load or train intent classifier
        if classifier_path and Path(classifier_path).exists():
            self._clf = IntentClassifier.load(classifier_path)
        else:
            self._clf = build_classifier(dataset_path)

    # ── public API ────────────────────────────

    def process(self, user_input: str, top_k: int = 1) -> dict:
        """
        Full pipeline for one user input.

        Returns:
        {
          "entrada_original":     str,
          "classificacao_intent": {label, rank, confidence, scores},
          "contexto_banco_dados": {comando, categoria, descricao,
                                   vector_search_distance},
          "validacao":            {trust_score, decision, signals, summary, ...},
          "decision":             "EXECUTE" | "CONFIRM" | "REJECT",
          "command":              str   # the command to run (if EXECUTE/CONFIRM)
        }
        """
        # 1. Classify intent
        intent = self._clf.predict(user_input)

        # 2. Vector search in command DB
        q_vec = _query_vector(user_input, self._vocab, self._idf, len(self._vocab))
        distances = [
            _cosine_distance(q_vec, dv)
            for dv in self._vecs
        ]
        best_idx  = min(range(len(distances)), key=lambda i: distances[i])
        best_cmd  = self._commands[best_idx]
        best_dist = distances[best_idx]

        db_context = {
            "comando":                best_cmd["comando"],
            "categoria":              best_cmd["categoria"],
            "descricao":              best_cmd["descricao"],
            "vector_search_distance": best_dist,
        }

        # 3. Build pipeline output (matches your existing JSON schema)
        pipeline_output = {
            "entrada_original":     user_input,
            "classificacao_intent": intent,
            "contexto_banco_dados": db_context,
        }

        # 4. Validate
        validation = validate(pipeline_output)

        # 5. Assemble result
        result = {
            **pipeline_output,
            "validacao": validation,
            "decision":  validation["decision"],
            "command":   best_cmd["comando"],
        }

        if self.verbose:
            self._print_result(result)

        return result

    def process_batch(self, inputs: list[str]) -> list[dict]:
        return [self.process(inp) for inp in inputs]

    # ── pretty printer ────────────────────────

    @staticmethod
    def _print_result(r: dict):
        v  = r["validacao"]
        sc = v["trust_score"]
        dec = v["decision"]
        bar = "█" * int(sc * 10) + "░" * (10 - int(sc * 10))
        print(f"\n{'─'*55}")
        print(f"  Input   : {r['entrada_original']}")
        print(f"  Intent  : {r['classificacao_intent']['label']}  "
              f"rank={r['classificacao_intent']['rank']}  "
              f"conf={r['classificacao_intent']['confidence']:.2f}")
        print(f"  Command : {r['command']}")
        print(f"  Trust   : [{bar}] {sc:.2f}  →  {dec}")
        print(f"  Summary : {v['summary']}")
        print("  Signals :")
        for s in v["signals"]:
            icon = {"ok": "✓", "warn": "⚠", "fail": "✗"}[s["status"]]
            print(f"    {icon} {s['name']:<25} {s['value']:.2f}  {s['reason']}")