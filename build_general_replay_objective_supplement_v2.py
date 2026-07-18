#!/usr/bin/env python3
"""Build the diversity-gated exact objective supplement for replay v1."""

from __future__ import annotations

import argparse
from collections import Counter
from math import gcd
import hashlib
import json
from pathlib import Path
from typing import Any

import build_general_replay_objective_supplement_v1 as v1
from build_general_replay_corpus_v1 import load_qwen_tokenizer
from general_replay_v1 import (
    ROOT,
    build_reference_compiler_rows,
    canonical_bytes,
    canonical_sha256,
    exact_token_group_subset,
    file_sha256,
    token_budgets,
    validate_prompt_specs,
)
from prepare_general_replay_candidate_shards_v1 import load_prompt_artifacts
from run_general_replay_candidate_shard_v1 import (
    MODEL_DIRECTORY,
    MODEL_FILE_SHA256,
    MODEL_REVISION,
)


DIRECTORY = ROOT / "data/general_replay_v1"
DEFAULT_SPEC_OUTPUT = DIRECTORY / "objective_supplement_prompt_specs_v2r1_150k.jsonl"
DEFAULT_ROW_OUTPUT = DIRECTORY / "objective_supplement_reference_rows_v2r1_150k.jsonl"
DEFAULT_REPORT = DIRECTORY / "objective_supplement_v2r1_150k.report.json"
DEFAULT_TARGET = 150_000
BUILD_SEED = 2026071802
ORDINAL_BASE = 200_000

SCOPED_CATEGORIES = v1.SCOPED_CATEGORIES
POOL_COUNTS = {
    "coding_debugging": 384,
    "mathematical_reasoning": 1_600,
    "json_structured_data": 240,
    "instruction_following": 960,
    "multilingual": 2_400,
}

FAMILY_GATES = {
    "coding_debugging": {"minimum_families": 12, "maximum_share_milli": 200},
    "mathematical_reasoning": {"minimum_families": 8, "maximum_share_milli": 250},
    "json_structured_data": {"minimum_families": 6, "maximum_share_milli": 400},
    "instruction_following": {"minimum_families": 8, "maximum_share_milli": 250},
    "multilingual": {"minimum_families": 96, "maximum_share_milli": 30},
}

BEFORE_DIVERSITY = {
    "coding_debugging": {"selected_rows": 52, "families": 6, "largest_share_milli": 192},
    "mathematical_reasoning": {"selected_rows": 576, "families": 1, "largest_share_milli": 1_000},
    "json_structured_data": {"selected_rows": 17, "families": 1, "largest_share_milli": 1_000},
    "instruction_following": {"selected_rows": 391, "families": 4, "largest_share_milli": 255},
    "multilingual": {
        "selected_rows": 845,
        "authored_pairs": 24,
        "semantic_families_per_language": 3,
        "largest_share_milli": 42,
    },
}


def _make_spec(category: str, index: int, prompt: str, verifier: dict,
               response_format: str, bounds: tuple[int, int]) -> dict:
    category_offset = sum(
        POOL_COUNTS[item]
        for item in SCOPED_CATEGORIES[:SCOPED_CATEGORIES.index(category)]
    )
    return v1._make_spec(
        category=category,
        ordinal=ORDINAL_BASE + category_offset + index,
        messages=v1._messages(prompt),
        verifier=verifier,
        response_format=response_format,
        bounds=bounds,
        build_seed=BUILD_SEED,
    )


CODING_FAMILIES = (
    "offset_map", "threshold_count", "divisible_filter", "running_totals",
    "cyclic_rotate", "stable_unique", "inclusive_range_debug",
    "accumulator_debug", "merge_sorted", "adjacent_differences",
    "fixed_chunks", "bounded_clamp",
)


def _coding_spec(index: int) -> dict:
    family_index = index % len(CODING_FAMILIES)
    variant = index // len(CODING_FAMILIES)
    family = CODING_FAMILIES[family_index]
    name = f"diverse_{family}_{variant}"
    if family == "offset_map":
        value = variant % 19 - 9
        requirement = f"add {value} to every integer in `values`, preserving order"
        cases = [{"args": [[-3, 0, 8]], "expected": [-3 + value, value, 8 + value]}, {"args": [[]], "expected": []}]
        code = f"def {name}(values):\n    return [item + {value} for item in values]"
    elif family == "threshold_count":
        value = variant % 23 - 11
        requirement = f"count integers in `values` strictly greater than {value}"
        cases = [{"args": [[value - 1, value, value + 1, value + 4]], "expected": 2}, {"args": [[]], "expected": 0}]
        code = f"def {name}(values):\n    return sum(item > {value} for item in values)"
    elif family == "divisible_filter":
        value = 2 + variant % 11
        requirement = f"return integers divisible by {value}, preserving order"
        cases = [{"args": [[value, value + 1, -2 * value]], "expected": [value, -2 * value]}, {"args": [[]], "expected": []}]
        code = f"def {name}(values):\n    return [item for item in values if item % {value} == 0]"
    elif family == "running_totals":
        requirement = "return the running total after each integer in `values`"
        cases = [{"args": [[3, -1, 4]], "expected": [3, 2, 6]}, {"args": [[]], "expected": []}]
        code = f"def {name}(values):\n    total = 0\n    result = []\n    for item in values:\n        total += item\n        result = result + [total]\n    return result"
    elif family == "cyclic_rotate":
        value = 1 + variant % 9
        requirement = f"rotate nonempty `values` left by {value} positions cyclically; return [] for empty input"
        source = [10, 20, 30, 40, 50]
        shift = value % len(source)
        cases = [{"args": [source], "expected": source[shift:] + source[:shift]}, {"args": [[]], "expected": []}]
        code = f"def {name}(values):\n    if not values:\n        return []\n    shift = {value} % len(values)\n    return values[shift:] + values[:shift]"
    elif family == "stable_unique":
        requirement = "return the first occurrence of each integer, preserving order"
        cases = [{"args": [[4, 2, 4, 3, 2]], "expected": [4, 2, 3]}, {"args": [[]], "expected": []}]
        code = f"def {name}(values):\n    result = []\n    for item in values:\n        if item not in result:\n            result = result + [item]\n    return result"
    elif family == "inclusive_range_debug":
        stop = 3 + variant % 13
        requirement = f"fix an off-by-one bug by returning every integer from 0 through {stop}, inclusive"
        cases = [{"args": [], "expected": list(range(stop + 1))}]
        code = f"def {name}():\n    return list(range({stop} + 1))"
    elif family == "accumulator_debug":
        requirement = "fix the accumulator logic and return the product of all integers; return 1 for empty input"
        cases = [{"args": [[2, -3, 4]], "expected": -24}, {"args": [[]], "expected": 1}]
        code = f"def {name}(values):\n    result = 1\n    for item in values:\n        result *= item\n    return result"
    elif family == "merge_sorted":
        requirement = "merge two ascending integer lists and return one ascending list"
        cases = [{"args": [[1, 4, 8], [2, 3, 9]], "expected": [1, 2, 3, 4, 8, 9]}, {"args": [[], []], "expected": []}]
        code = f"def {name}(left, right):\n    return sorted(left + right)"
    elif family == "adjacent_differences":
        requirement = "return each later integer minus its predecessor; inputs shorter than two return []"
        cases = [{"args": [[3, 8, 6, 10]], "expected": [5, -2, 4]}, {"args": [[1]], "expected": []}]
        code = f"def {name}(values):\n    return [values[index] - values[index - 1] for index in range(1, len(values))]"
    elif family == "fixed_chunks":
        width = 2 + variant % 5
        requirement = f"split `values` into consecutive chunks of width {width}, retaining a shorter final chunk"
        cases = [{"args": [[1, 2, 3, 4, 5]], "expected": [[1, 2, 3, 4, 5][i:i + width] for i in range(0, 5, width)]}, {"args": [[]], "expected": []}]
        code = f"def {name}(values):\n    return [values[index:index + {width}] for index in range(0, len(values), {width})]"
    else:
        low = -(2 + variant % 7)
        high = 4 + variant % 11
        requirement = f"limit every integer to the inclusive range [{low}, {high}]"
        cases = [{"args": [[low - 3, 0, high + 3]], "expected": [low, 0, high]}, {"args": [[]], "expected": []}]
        code = f"def {name}(values):\n    return [min({high}, max({low}, item)) for item in values]"
    prompt = (
        f"Implement `{name}` in Python: {requirement}. Return one Python code "
        "block containing only the function; use no imports or global state."
    )
    return _make_spec(
        "coding_debugging", index, prompt,
        {"type": "python_function_cases_v1", "status": "ready", "config": {
            "function_name": name, "cases": cases,
            "reference_answer": f"```python\n{code}\n```", "timeout_seconds": 2,
        }},
        "assistant_text", (1, 384),
    )


MATH_FAMILIES = (
    "linear_equation", "fraction_sum", "modular_arithmetic", "rectangle_geometry",
    "arithmetic_sequence", "finite_probability", "two_variable_system",
    "integer_mean",
)


def _math_spec(index: int) -> dict:
    family = MATH_FAMILIES[index % len(MATH_FAMILIES)]
    variant = index // len(MATH_FAMILIES)
    if family == "linear_equation":
        a = 2 + variant % 13; x = variant - 90; b = variant * 7 - 31; result = a * x + b
        prompt = f"Solve {a}x + ({b}) = {result}. Return only JSON with integer keys `x` and `check`."
        expected = {"check": result, "x": x}
    elif family == "fraction_sum":
        a = 1 + variant % 9; b = 2 + variant % 11; c = 1 + (variant * 3) % 8; d = 3 + variant % 13
        numerator = a * d + c * b; denominator = b * d; divisor = gcd(abs(numerator), denominator)
        expected = {"denominator": denominator // divisor, "numerator": numerator // divisor}
        prompt = f"Add {a}/{b} and {c}/{d}. Return only JSON with the reduced integer `numerator` and positive `denominator`."
    elif family == "modular_arithmetic":
        a = 17 + variant * 3; b = 5 + variant % 19; c = 7 + variant % 23; modulus = 11 + variant % 29
        expected = {"remainder": (a * b + c) % modulus}
        prompt = f"Compute ({a} * {b} + {c}) modulo {modulus}. Return only JSON with integer key `remainder`."
    elif family == "rectangle_geometry":
        width = 2 + variant % 31; height = 3 + (variant * 5) % 29
        expected = {"area": width * height, "perimeter": 2 * (width + height)}
        prompt = f"A rectangle is {width} units wide and {height} units high. Return only JSON with integer `area` and `perimeter`."
    elif family == "arithmetic_sequence":
        first = variant % 17 - 8; step = 2 + variant % 9; n = 5 + variant % 24
        nth = first + (n - 1) * step; total = n * (first + nth) // 2
        expected = {"nth": nth, "sum": total}
        prompt = f"An arithmetic sequence starts at {first} with common difference {step}. Return only JSON for term {n} using integer keys `nth` and `sum` (sum of first {n} terms)."
    elif family == "finite_probability":
        favorable = 1 + variant % 9; other = 2 + variant * 5; total = favorable + other; divisor = gcd(favorable, total)
        expected = {"denominator": total // divisor, "numerator": favorable // divisor}
        prompt = f"A bag has {favorable} blue and {other} amber tokens. One is drawn uniformly. Return only JSON for P(blue) as reduced integer `numerator` and `denominator`."
    elif family == "two_variable_system":
        x = variant % 41 - 20; y = (variant * 3) % 37 - 18; a = 2 + variant % 7; b = 2 + (variant * 5) % 9
        first = a * x + y; second = x - b * y
        expected = {"x": x, "y": y}
        prompt = f"Solve the system {a}x + y = {first} and x - {b}y = {second}. Return only JSON with integer keys `x` and `y`."
    else:
        base = variant - 100; values = [base, base + 2, base + 4, base + 6]
        expected = {"mean": sum(values) // len(values), "sum": sum(values)}
        prompt = f"For the integers {values}, return only JSON with integer keys `sum` and `mean`."
    return _make_spec(
        "mathematical_reasoning", index, prompt,
        {"type": "json_exact_v1", "status": "ready", "config": {"expected": expected}},
        "assistant_json", (1, 160),
    )


JSON_FAMILIES = (
    "summary", "threshold_filter", "group_totals", "id_sort",
    "stable_tags", "field_projection",
)


def _json_spec(index: int) -> dict:
    family = JSON_FAMILIES[index % len(JSON_FAMILIES)]
    variant = index // len(JSON_FAMILIES)
    records = [
        {"id": f"J{variant:03d}-{position}", "group": ("amber", "blue", "cedar")[(variant + position) % 3], "value": 2 + (variant * 11 + position * 7) % 41}
        for position in range(3 + variant % 4)
    ]
    if family == "summary":
        expected = {"count": len(records), "total": sum(r["value"] for r in records)}
        task = "Return `count` and the sum as `total`."
    elif family == "threshold_filter":
        threshold = 18 + variant % 9; expected = {"records": [r for r in records if r["value"] >= threshold]}
        task = f"Return `records` containing only rows with value at least {threshold}, preserving order."
    elif family == "group_totals":
        totals = {}
        for record in records: totals[record["group"]] = totals.get(record["group"], 0) + record["value"]
        expected = {"totals": dict(sorted(totals.items()))}; task = "Return `totals` by group with alphabetically sorted keys."
    elif family == "id_sort":
        source = list(reversed(records)); records = source; expected = {"records": sorted(source, key=lambda r: r["id"])}
        task = "Return all `records` sorted by id, preserving every field."
    elif family == "stable_tags":
        tags = [r["group"] for r in records] + [records[0]["group"]]; seen = []
        for tag in tags:
            if tag not in seen: seen.append(tag)
        expected = {"tags": seen}; records = [{"case": f"JT-{variant:04d}", "tags": tags}]; task = "Return `tags` with duplicates removed in first-seen order; ignore `case`."
    else:
        expected = {"records": [{"id": r["id"], "value": r["value"]} for r in records]}; task = "Return `records` projected to only `id` and `value`, preserving order."
    prompt = task + " Return only JSON. Input: " + json.dumps(records, sort_keys=True)
    return _make_spec(
        "json_structured_data", index, prompt,
        {"type": "json_exact_v1", "status": "ready", "config": {"expected": expected}},
        "assistant_json", (1, 256),
    )


INSTRUCTION_FAMILIES = (
    "two_lines", "status_line", "ready_line", "four_lines",
    "csv_line", "pipe_line", "three_bullets_plain", "tag_pair",
)


def _instruction_spec(index: int) -> dict:
    family = INSTRUCTION_FAMILIES[index % len(INSTRUCTION_FAMILIES)]
    number = 500 + index
    if family == "two_lines": expected = f"ALPHA: {number}\nBETA: {number + 7}"
    elif family == "status_line": expected = f"status=ready; batch={number}; retries=0"
    elif family == "ready_line": expected = f"READY: {number}"
    elif family == "four_lines": expected = f"ITEM {number}\nOWNER operations\nSTATE queued\nPRIORITY normal"
    elif family == "csv_line": expected = f"{number},ready,normal"
    elif family == "pipe_line": expected = f"job-{number}|queued|0"
    elif family == "three_bullets_plain": expected = f"- red-{number}\n- green-{number + 1}\n- blue-{number + 2}"
    else: expected = f"<code>{number}</code><state>ready</state>"
    prompt = f"Return exactly the following text, preserving punctuation and line breaks, with no surrounding commentary or code fence:\n{expected}"
    return _make_spec(
        "instruction_following", index, prompt,
        {"type": "exact_text_v1", "status": "ready", "config": {"expected": expected}},
        "assistant_text", (1, 160),
    )


# Twelve independently authored semantic sentence families in each of eight
# languages. Only opaque code/room slots vary; all lexical choices are fixed.
TRANSLATION_TEMPLATES = (
    ("Spanish", "Write code {code} on the blue card.", "Escribe el código {code} en la tarjeta azul."),
    ("Spanish", "Package {code} is beside the window.", "El paquete {code} está junto a la ventana."),
    ("Spanish", "Please take folder {code} to room {room}.", "Por favor, lleva la carpeta {code} a la sala {room}."),
    ("Spanish", "Meeting {code} starts tomorrow at nine.", "La reunión {code} empieza mañana a las nueve."),
    ("Spanish", "Save document {code} before closing the window.", "Guarda el documento {code} antes de cerrar la ventana."),
    ("Spanish", "Library room {room} is quiet on Sunday afternoon.", "La sala {room} de la biblioteca está tranquila el domingo por la tarde."),
    ("Spanish", "We need three clean cups for table {code}.", "Necesitamos tres tazas limpias para la mesa {code}."),
    ("Spanish", "Train {code} arrives five minutes after noon.", "El tren {code} llega cinco minutos después del mediodía."),
    ("Spanish", "Keys {code} are next to the green notebook.", "Las llaves {code} están junto al cuaderno verde."),
    ("Spanish", "Blue bicycle {code} belongs to my neighbor.", "La bicicleta azul {code} pertenece a mi vecino."),
    ("Spanish", "Put red box {code} under the chair in room {room}.", "Pon la caja roja {code} debajo de la silla en la sala {room}."),
    ("Spanish", "Turn off the hallway light near room {room} before leaving.", "Apaga la luz del pasillo cerca de la sala {room} antes de salir."),
    ("French", "Write code {code} on the blue card.", "Écrivez le code {code} sur la carte bleue."),
    ("French", "Package {code} is beside the window.", "Le colis {code} est à côté de la fenêtre."),
    ("French", "Please take folder {code} to room {room}.", "Veuillez apporter le dossier {code} à la salle {room}."),
    ("French", "Meeting {code} starts tomorrow at nine.", "La réunion {code} commence demain à neuf heures."),
    ("French", "Save document {code} before closing the window.", "Enregistrez le document {code} avant de fermer la fenêtre."),
    ("French", "Library room {room} is quiet on Sunday afternoon.", "La salle {room} de la bibliothèque est calme le dimanche après-midi."),
    ("French", "We need three clean cups for table {code}.", "Nous avons besoin de trois tasses propres pour la table {code}."),
    ("French", "Train {code} arrives five minutes after noon.", "Le train {code} arrive cinq minutes après midi."),
    ("French", "Keys {code} are next to the green notebook.", "Les clés {code} sont à côté du carnet vert."),
    ("French", "Blue bicycle {code} belongs to my neighbor.", "Le vélo bleu {code} appartient à mon voisin."),
    ("French", "Put red box {code} under the chair in room {room}.", "Mettez la boîte rouge {code} sous la chaise dans la salle {room}."),
    ("French", "Turn off the hallway light near room {room} before leaving.", "Éteignez la lumière du couloir près de la salle {room} avant de partir."),
    ("German", "Write code {code} on the blue card.", "Schreiben Sie den Code {code} auf die blaue Karte."),
    ("German", "Package {code} is beside the window.", "Das Paket {code} liegt neben dem Fenster."),
    ("German", "Please take folder {code} to room {room}.", "Bitte bringen Sie den Ordner {code} in Raum {room}."),
    ("German", "Meeting {code} starts tomorrow at nine.", "Die Besprechung {code} beginnt morgen um neun Uhr."),
    ("German", "Save document {code} before closing the window.", "Speichern Sie das Dokument {code}, bevor Sie das Fenster schließen."),
    ("German", "Library room {room} is quiet on Sunday afternoon.", "Raum {room} der Bibliothek ist am Sonntagnachmittag ruhig."),
    ("German", "We need three clean cups for table {code}.", "Wir brauchen drei saubere Tassen für den Tisch {code}."),
    ("German", "Train {code} arrives five minutes after noon.", "Der Zug {code} kommt fünf Minuten nach zwölf an."),
    ("German", "Keys {code} are next to the green notebook.", "Die Schlüssel {code} liegen neben dem grünen Notizbuch."),
    ("German", "Blue bicycle {code} belongs to my neighbor.", "Das blaue Fahrrad {code} gehört meinem Nachbarn."),
    ("German", "Put red box {code} under the chair in room {room}.", "Stellen Sie die rote Kiste {code} unter den Stuhl in Raum {room}."),
    ("German", "Turn off the hallway light near room {room} before leaving.", "Schalten Sie das Flurlicht bei Raum {room} aus, bevor Sie gehen."),
    ("Portuguese", "Write code {code} on the blue card.", "Escreva o código {code} no cartão azul."),
    ("Portuguese", "Package {code} is beside the window.", "O pacote {code} está ao lado da janela."),
    ("Portuguese", "Please take folder {code} to room {room}.", "Por favor, leve a pasta {code} para a sala {room}."),
    ("Portuguese", "Meeting {code} starts tomorrow at nine.", "A reunião {code} começa amanhã às nove."),
    ("Portuguese", "Save document {code} before closing the window.", "Salve o documento {code} antes de fechar a janela."),
    ("Portuguese", "Library room {room} is quiet on Sunday afternoon.", "A sala {room} da biblioteca fica silenciosa no domingo à tarde."),
    ("Portuguese", "We need three clean cups for table {code}.", "Precisamos de três xícaras limpas para a mesa {code}."),
    ("Portuguese", "Train {code} arrives five minutes after noon.", "O trem {code} chega cinco minutos depois do meio-dia."),
    ("Portuguese", "Keys {code} are next to the green notebook.", "As chaves {code} estão ao lado do caderno verde."),
    ("Portuguese", "Blue bicycle {code} belongs to my neighbor.", "A bicicleta azul {code} pertence ao meu vizinho."),
    ("Portuguese", "Put red box {code} under the chair in room {room}.", "Coloque a caixa vermelha {code} debaixo da cadeira na sala {room}."),
    ("Portuguese", "Turn off the hallway light near room {room} before leaving.", "Apague a luz do corredor perto da sala {room} antes de sair."),
    ("Italian", "Write code {code} on the blue card.", "Scriva il codice {code} sulla scheda blu."),
    ("Italian", "Package {code} is beside the window.", "Il pacco {code} è accanto alla finestra."),
    ("Italian", "Please take folder {code} to room {room}.", "Per favore, porti la cartella {code} nella stanza {room}."),
    ("Italian", "Meeting {code} starts tomorrow at nine.", "La riunione {code} inizia domani alle nove."),
    ("Italian", "Save document {code} before closing the window.", "Salva il documento {code} prima di chiudere la finestra."),
    ("Italian", "Library room {room} is quiet on Sunday afternoon.", "La sala {room} della biblioteca è tranquilla la domenica pomeriggio."),
    ("Italian", "We need three clean cups for table {code}.", "Ci servono tre tazze pulite per il tavolo {code}."),
    ("Italian", "Train {code} arrives five minutes after noon.", "Il treno {code} arriva cinque minuti dopo mezzogiorno."),
    ("Italian", "Keys {code} are next to the green notebook.", "Le chiavi {code} sono accanto al quaderno verde."),
    ("Italian", "Blue bicycle {code} belongs to my neighbor.", "La bicicletta blu {code} appartiene al mio vicino."),
    ("Italian", "Put red box {code} under the chair in room {room}.", "Metti la scatola rossa {code} sotto la sedia nella stanza {room}."),
    ("Italian", "Turn off the hallway light near room {room} before leaving.", "Spegni la luce del corridoio vicino alla stanza {room} prima di uscire."),
    ("Japanese", "Write code {code} on the blue card.", "青いカードにコード{code}を書いてください。"),
    ("Japanese", "Package {code} is beside the window.", "荷物{code}は窓のそばにあります。"),
    ("Japanese", "Please take folder {code} to room {room}.", "フォルダー{code}を{room}号室に持って行ってください。"),
    ("Japanese", "Meeting {code} starts tomorrow at nine.", "会議{code}は明日の午前9時に始まります。"),
    ("Japanese", "Save document {code} before closing the window.", "窓を閉じる前に文書{code}を保存してください。"),
    ("Japanese", "Library room {room} is quiet on Sunday afternoon.", "図書館の{room}号室は日曜日の午後は静かです。"),
    ("Japanese", "We need three clean cups for table {code}.", "テーブル{code}にはきれいなカップが3つ必要です。"),
    ("Japanese", "Train {code} arrives five minutes after noon.", "列車{code}は正午の5分後に到着します。"),
    ("Japanese", "Keys {code} are next to the green notebook.", "鍵{code}は緑色のノートの隣にあります。"),
    ("Japanese", "Blue bicycle {code} belongs to my neighbor.", "青い自転車{code}は私の隣人のものです。"),
    ("Japanese", "Put red box {code} under the chair in room {room}.", "赤い箱{code}を{room}号室の椅子の下に置いてください。"),
    ("Japanese", "Turn off the hallway light near room {room} before leaving.", "帰る前に{room}号室の近くの廊下の明かりを消してください。"),
    ("Dutch", "Write code {code} on the blue card.", "Schrijf code {code} op de blauwe kaart."),
    ("Dutch", "Package {code} is beside the window.", "Pakket {code} ligt naast het raam."),
    ("Dutch", "Please take folder {code} to room {room}.", "Breng map {code} alstublieft naar kamer {room}."),
    ("Dutch", "Meeting {code} starts tomorrow at nine.", "Vergadering {code} begint morgen om negen uur."),
    ("Dutch", "Save document {code} before closing the window.", "Sla document {code} op voordat u het venster sluit."),
    ("Dutch", "Library room {room} is quiet on Sunday afternoon.", "Zaal {room} van de bibliotheek is op zondagmiddag stil."),
    ("Dutch", "We need three clean cups for table {code}.", "We hebben drie schone kopjes nodig voor tafel {code}."),
    ("Dutch", "Train {code} arrives five minutes after noon.", "Trein {code} arriveert vijf minuten na het middaguur."),
    ("Dutch", "Keys {code} are next to the green notebook.", "Sleutels {code} liggen naast het groene notitieboek."),
    ("Dutch", "Blue bicycle {code} belongs to my neighbor.", "De blauwe fiets {code} is van mijn buurman."),
    ("Dutch", "Put red box {code} under the chair in room {room}.", "Zet de rode doos {code} onder de stoel in kamer {room}."),
    ("Dutch", "Turn off the hallway light near room {room} before leaving.", "Doe het licht in de gang bij kamer {room} uit voordat u vertrekt."),
    ("Swedish", "Write code {code} on the blue card.", "Skriv koden {code} på det blå kortet."),
    ("Swedish", "Package {code} is beside the window.", "Paketet {code} ligger bredvid fönstret."),
    ("Swedish", "Please take folder {code} to room {room}.", "Ta mappen {code} till rum {room}, tack."),
    ("Swedish", "Meeting {code} starts tomorrow at nine.", "Mötet {code} börjar klockan nio i morgon."),
    ("Swedish", "Save document {code} before closing the window.", "Spara dokumentet {code} innan du stänger fönstret."),
    ("Swedish", "Library room {room} is quiet on Sunday afternoon.", "Bibliotekets rum {room} är tyst på söndag eftermiddag."),
    ("Swedish", "We need three clean cups for table {code}.", "Vi behöver tre rena koppar till bordet {code}."),
    ("Swedish", "Train {code} arrives five minutes after noon.", "Tåg {code} anländer fem minuter efter klockan tolv."),
    ("Swedish", "Keys {code} are next to the green notebook.", "Nycklarna {code} ligger bredvid den gröna anteckningsboken."),
    ("Swedish", "Blue bicycle {code} belongs to my neighbor.", "Den blå cykeln {code} tillhör min granne."),
    ("Swedish", "Put red box {code} under the chair in room {room}.", "Ställ den röda lådan {code} under stolen i rum {room}."),
    ("Swedish", "Turn off the hallway light near room {room} before leaving.", "Släck lampan i korridoren nära rum {room} innan du går."),
)


def _multilingual_spec(index: int) -> dict:
    language, source_template, target_template = TRANSLATION_TEMPLATES[index % len(TRANSLATION_TEMPLATES)]
    code = f"{chr(65 + index % 26)}{chr(65 + (index // 26) % 26)}-{2_000 + index}"
    room = str(100 + index % 800)
    source = source_template.format(code=code, room=room)
    target = target_template.format(code=code, room=room)
    prompt = f"Translate into {language}. Return only the translation: {source}"
    return _make_spec(
        "multilingual", index, prompt,
        {"type": "exact_text_v1", "status": "ready", "config": {"expected": target}},
        "assistant_text", (1, 160),
    )


POOL_BUILDERS = {
    "coding_debugging": _coding_spec,
    "mathematical_reasoning": _math_spec,
    "json_structured_data": _json_spec,
    "instruction_following": _instruction_spec,
    "multilingual": _multilingual_spec,
}
FAMILIES = {
    "coding_debugging": CODING_FAMILIES,
    "mathematical_reasoning": MATH_FAMILIES,
    "json_structured_data": JSON_FAMILIES,
    "instruction_following": INSTRUCTION_FAMILIES,
    "multilingual": tuple(f"{language}:{index % 12:02d}" for index, (language, _, _) in enumerate(TRANSLATION_TEMPLATES)),
}


def build_candidate_pool() -> tuple[list[dict], dict[str, str]]:
    specs = []
    family_by_spec = {}
    for category in SCOPED_CATEGORIES:
        names = FAMILIES[category]
        for index in range(POOL_COUNTS[category]):
            spec = POOL_BUILDERS[category](index)
            specs.append(spec)
            family_by_spec[spec["spec_id"]] = names[index % len(names)]
    validate_prompt_specs(specs)
    if any(spec["verifier"]["type"] == "approval_required_v1" for spec in specs):
        raise AssertionError("diverse supplement contains a subjective gate")
    return specs, family_by_spec


def diversity_audit(selected_rows: list[dict], family_by_spec: dict[str, str]) -> dict:
    result = {}
    for category in SCOPED_CATEGORIES:
        rows = [row for row in selected_rows if row["category"] == category]
        counts = Counter(family_by_spec[row["lineage"]["spec_id"]] for row in rows)
        largest = max(counts.values(), default=0)
        gate = FAMILY_GATES[category]
        item = {
            "selected_rows": len(rows),
            "represented_families": len(counts),
            "family_counts": dict(sorted(counts.items())),
            "largest_family_rows": largest,
            "largest_family_share_milli": (
                largest * 1_000 // len(rows) if rows else 0
            ),
            "minimum_families_required": gate["minimum_families"],
            "maximum_family_share_milli": gate["maximum_share_milli"],
        }
        item["passes"] = (
            item["represented_families"] >= gate["minimum_families"]
            and item["largest_family_share_milli"] <= gate["maximum_share_milli"]
        )
        result[category] = item
    if not all(item["passes"] for item in result.values()):
        raise RuntimeError("selected objective supplement fails diversity gates")
    return result


def select_exact(pool_specs: list[dict], tokenizer: Any, required: dict[str, int],
                 family_by_spec: dict[str, str]) -> tuple[list[dict], list[dict], dict]:
    pool_rows, compiler_audit = build_reference_compiler_rows(pool_specs, tokenizer)
    selected_rows = []
    category_audit = {}
    for category in SCOPED_CATEGORIES:
        pool = [row for row in pool_rows if row["category"] == category]
        selected = exact_token_group_subset(pool, required[category])
        selected_rows.extend(selected)
        category_audit[category] = {
            "pool_specs": POOL_COUNTS[category],
            "pool_assistant_tokens": sum(row["assistant_token_count"] for row in pool),
            "required_supplement_tokens": required[category],
            "selected_rows": len(selected),
            "selected_assistant_tokens": sum(row["assistant_token_count"] for row in selected),
            "exact_unique_source_group_subset_reached": True,
        }
    selected_ids = {row["lineage"]["spec_id"] for row in selected_rows}
    selected_specs = [spec for spec in pool_specs if spec["spec_id"] in selected_ids]
    selected_specs.sort(key=lambda item: item["spec_id"])
    selected_rows.sort(key=lambda item: item["row_id"])
    validate_prompt_specs(selected_specs)
    return selected_specs, selected_rows, {
        "compiler": compiler_audit,
        "categories": category_audit,
        "diversity": diversity_audit(selected_rows, family_by_spec),
    }


def build_artifacts(*, base_specs: list[dict], base_rows: list[dict],
                    base_row_report: dict, tokenizer: Any, target: int):
    if target != DEFAULT_TARGET:
        raise ValueError("diverse objective supplement v2 is sealed to 150k")
    budgets = token_budgets(target)
    seed_tokens = v1._seed_token_counts(tokenizer)
    base_tokens = Counter()
    for row in base_rows: base_tokens[row["category"]] += row["assistant_token_count"]
    required = {category: budgets[category] - seed_tokens[category] - base_tokens[category] for category in SCOPED_CATEGORIES}
    pool_specs, family_by_spec = build_candidate_pool()
    selected_specs, selected_rows, selection = select_exact(pool_specs, tokenizer, required, family_by_spec)
    validate_prompt_specs([*base_specs, *selected_specs])
    supplement_tokens = Counter()
    for row in selected_rows: supplement_tokens[row["category"]] += row["assistant_token_count"]
    accounting = {
        category: {
            "target_assistant_tokens": budgets[category],
            "mandatory_seed_tokens": seed_tokens[category],
            "base_reference_tokens": base_tokens[category],
            "supplement_reference_tokens": supplement_tokens[category],
            "realized_assistant_tokens": seed_tokens[category] + base_tokens[category] + supplement_tokens[category],
            "exact": seed_tokens[category] + base_tokens[category] + supplement_tokens[category] == budgets[category],
        }
        for category in SCOPED_CATEGORIES
    }
    if not all(item["exact"] for item in accounting.values()):
        raise AssertionError("diverse supplement accounting is not exact")
    spec_bytes = b"".join(canonical_bytes(item) for item in selected_specs)
    row_bytes = b"".join(canonical_bytes(item) for item in selected_rows)
    report = {
        "schema": "general-replay-objective-supplement-report-v2",
        "revision": "v2r1",
        "status": "exact_scoped_category_capacity_and_diversity_ready",
        "target_total_assistant_tokens": target,
        "scoped_categories": list(SCOPED_CATEGORIES),
        "build_seed": BUILD_SEED,
        "ordinal_base": ORDINAL_BASE,
        "base_reference": {
            "output_sha256": base_row_report["output_sha256"],
            "rows": len(base_rows),
            "report_content_sha256": base_row_report["content_sha256_before_self_field"],
        },
        "supersedes": {
            "schema": "general-replay-objective-supplement-report-v1",
            "reason": "exact-token artifact failed semantic diversity review",
            "before_diversity": BEFORE_DIVERSITY,
        },
        "outputs": {
            "prompt_specs": {"rows": len(selected_specs), "sha256": hashlib.sha256(spec_bytes).hexdigest()},
            "reference_rows": {"rows": len(selected_rows), "assistant_tokens": sum(supplement_tokens.values()), "sha256": hashlib.sha256(row_bytes).hexdigest()},
        },
        "selection": selection,
        "accounting": accounting,
        "all_scoped_category_targets_exact": True,
        "all_diversity_gates_pass": True,
        "translation_design": {
            "languages": 8,
            "hand_authored_pairs": len(TRANSLATION_TEMPLATES),
            "semantic_families_per_language": 12,
            "variable_fields": ["opaque_code", "room_number"],
            "unverified_generation_used": False,
        },
        "tokenizer": {
            "path": str(MODEL_DIRECTORY), "revision": MODEL_REVISION,
            "chat_template_sha256": MODEL_FILE_SHA256["chat_template.jinja"],
            "assistant_mask_policy": "official_qwen_assistant_only_v1",
        },
        "policy": {
            "deterministic_synthetic_sources_only": True,
            "direct_benchmark_prompts": False,
            "protected_or_evaluation_sources_opened": False,
            "subjective_rows_compiled": False,
            "same_objective_verifiers_reapplied": True,
            "duplicate_or_padding_fill_used": False,
            "one_row_per_unique_source_group": True,
            "base_generations_promoted": False,
        },
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return spec_bytes, row_bytes, report_bytes, report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-specs", type=Path, default=v1.DEFAULT_BASE_SPECS)
    result.add_argument("--base-spec-report", type=Path, default=v1.DEFAULT_BASE_SPEC_REPORT)
    result.add_argument("--base-rows", type=Path, default=v1.DEFAULT_BASE_ROWS)
    result.add_argument("--base-row-report", type=Path, default=v1.DEFAULT_BASE_ROW_REPORT)
    result.add_argument("--target-assistant-tokens", type=int, default=DEFAULT_TARGET)
    result.add_argument("--spec-output", type=Path, default=DEFAULT_SPEC_OUTPUT)
    result.add_argument("--row-output", type=Path, default=DEFAULT_ROW_OUTPUT)
    result.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    base_specs, base_spec_report = load_prompt_artifacts(args.base_specs, args.base_spec_report)
    base_rows, base_row_report = v1._load_base_rows(args.base_rows, args.base_row_report)
    if base_spec_report["output_sha256"] != file_sha256(args.base_specs):
        raise RuntimeError("base prompt artifact identity changed")
    tokenizer = load_qwen_tokenizer(str(MODEL_DIRECTORY), MODEL_REVISION, MODEL_FILE_SHA256["chat_template.jinja"])
    spec_bytes, row_bytes, report_bytes, report = build_artifacts(
        base_specs=base_specs, base_rows=base_rows,
        base_row_report=base_row_report, tokenizer=tokenizer,
        target=args.target_assistant_tokens,
    )
    outputs = (args.spec_output, args.row_output, args.report)
    expected = (spec_bytes, row_bytes, report_bytes)
    if args.check:
        if any(path.read_bytes() != content for path, content in zip(outputs, expected)):
            raise RuntimeError("diverse objective supplement artifacts changed")
        return 0
    if any(path.exists() for path in outputs):
        raise FileExistsError("diverse supplement build requires fresh outputs")
    for path in outputs: path.parent.mkdir(parents=True, exist_ok=True)
    for path, content in zip(outputs, expected): path.write_bytes(content)
    print(json.dumps({
        "spec_output": str(args.spec_output), "spec_sha256": report["outputs"]["prompt_specs"]["sha256"],
        "row_output": str(args.row_output), "row_sha256": report["outputs"]["reference_rows"]["sha256"],
        "report": str(args.report), "report_content_sha256": report["content_sha256_before_self_field"],
        "rows": report["outputs"]["reference_rows"]["rows"],
        "assistant_tokens": report["outputs"]["reference_rows"]["assistant_tokens"],
        "status": report["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
