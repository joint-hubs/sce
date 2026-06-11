# SCE 1.0.0 Release Plan (junior-engineer playbook)

**Created:** 2026-04-18
**Target version:** `1.0.0`
**Current version:** `0.3.5` (PyPI: `stat-context`)
**Owner:** junior dev (implementation), tech lead (sign-off, tag push, PyPI auth)
**Companion docs:** [2026-04-18_leakage_safe_remediation_plan.md](./2026-04-18_leakage_safe_remediation_plan.md)

---

## 0. Jak korzystać z tego dokumentu

1. Bloki **R0 → R5** mają być realizowane sekwencyjnie. R0 to twardy blocker — bez ukończenia bloku „leakage-safe" (osobny plan) nie wypuszczamy 1.0.0.
2. Każde zadanie = 1 PR. Tytuł: `[release-1.0][R{n}-{nr}] krótki opis`.
3. Punkty `⚠ DECYZJA WYMAGANA` muszą być wyjaśnione z leadem **przed** rozpoczęciem zadania.
4. Po każdym ukończonym zadaniu odhacz w [§9 tracking checklist](#9-tracking-checklist).
5. Tag `v1.0.0` pushuje **tylko** lead — junior nie ma do tego uprawnień.

---

## 1. Audyt obecnego stanu repo (snapshot 2026-04-18)

### Co jest OK ✅

| Obszar | Stan |
|---|---|
| `pyproject.toml` | PEP 621, klasyfikatory, optional deps, scripts entry-point |
| Type hints marker | `sce/py.typed` istnieje |
| CI matrix | Python 3.9–3.12 w [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) |
| Release flow | TestPyPI → PyPI w [`.github/workflows/release.yml`](../../.github/workflows/release.yml), trusted publishing OIDC |
| Lint/format | `ruff` + `black` skonfigurowane |
| Tests | pytest z coverage; 17+ plików testowych |
| Module headers | wszystkie pliki `sce/*.py` mają `@module/@depends/@exports/@paper_ref` |
| Docs site | MkDocs Material + GitHub Pages (`mkdocs.yml`) |
| MANIFEST.in | wyklucza data parquety, build artifacts |
| Changelog | format Keep a Changelog, semver |
| Citation | `CITATION.cff` istnieje |

### Co wymaga naprawy ⚠

| # | Problem | Severity | Detail |
|---|---|:---:|---|
| 1 | **Licencja `CC-BY-NC-4.0` nie jest OSI-approved** dla biblioteki software | 🔴 **blocker** | Non-Commercial blokuje użycie w jakiejkolwiek firmie. Większość detektorów (PyPI, GitHub) ostrzeże. Decyzja vs paper licensing. |
| 2 | **Heavy ML deps są required**, nie optional | 🔴 blocker | `xgboost`, `lightgbm`, `catboost`, `pyarrow` w `dependencies` — instalacja `pip install stat-context` ściąga ~500 MB. Powinny być optional extras. |
| 3 | **DeprecationWarnings** dla starego API | 🔴 blocker (1.0) | `ContextConfig.hierarchy`, `additional_categorical_cols`, `compute_aggregations(hierarchy=...)`, `include_quantiles` — w 1.0 usuwamy. |
| 4 | Naming inconsistency: `pip install stat-context`, `import sce` | 🟡 ważne | OK technicznie, ale myli użytkowników. Dokumentacja musi jawnie ostrzegać. |
| 5 | Dwa egg-info w repo: `sce.egg-info/`, `stat_context.egg-info/` | 🟡 ważne | Pozostałości po renamie. Wyrzucić, dodać do `.gitignore`. |
| 6 | `build/`, `dist/`, `htmlcov/`, `catboost_info/`, `experiment_debug.log`, `.release-smoke/` w repo | 🟡 ważne | Build artifacts nie powinny być commitowane. Sprawdzić `.gitignore`. |
| 7 | **Python 3.9 EOL** (październik 2025) | 🟡 ważne | Min 3.10 dla 1.0 to czysta opcja. Decyzja vs userbase. |
| 8 | `selection.py` test coverage 17% — w docstring jawnie „EXPERIMENTAL" | 🟡 ważne | Albo ukryć z public API w 1.0, albo dopisać testy. |
| 9 | `__init__.py` eksportuje 25+ symboli, w tym wewnętrzne | 🟡 ważne | Public API dla 1.0 musi być stabilne i minimalne. |
| 10 | Brak `SECURITY.md` | 🟢 nice-to-have | Standard dla 1.0 OSS. |
| 11 | Brak smoke test post-install w CI | 🟢 nice-to-have | Release.yml ma `import` test, ale brak end-to-end mini-pipeline. |
| 12 | Brak SBOM / wheel signing | 🟢 nice-to-have | PyPI trusted publishing załatwia attestations, ale można dodać `cibuildwheel` jeśli chcemy wheels per-platform (na razie pure Python — nie trzeba). |
| 13 | Linki w README mogą się rozjechać po publikacji na PyPI | 🟡 ważne | `0.3.5` historycznie miał ten problem (changelog). Sprawdzić wszystkie. |
| 14 | `print()` w bibliotecznym kodzie (`sce/engine.py` L120, L122) | 🟡 ważne | Library nigdy nie powinna `print` — tylko `logging`. |
| 15 | Brak min-supported-version testu (np. `numpy==1.21.0`, najstarsze pinie) | 🟢 nice-to-have | Dodać job w CI. |
| 16 | **Leakage-safe issues** (osobny plan) | 🔴 **HARD BLOCKER** | 1.0 nie może wyjść z otwartymi P0 z [`2026-04-18_leakage_safe_remediation_plan.md`](./2026-04-18_leakage_safe_remediation_plan.md). |

---

## 2. Filozofia 1.0

`1.0.0` w semver to **publiczne zobowiązanie do stabilnego API**. Każda przyszła breaking change wymaga 2.0. Konsekwencje:

- **Public API musi być świadomie wybrane.** Symbole eksportowane z `sce/__init__.py` zostają — ich modyfikacja w 1.x = breaking.
- **Deprecated API znika.** Wszystkie `DeprecationWarning` z 0.x są usuwane w 1.0.
- **Hard contracts:** wersje deps w pyproject muszą mieć górne ograniczenia tam, gdzie API rzeczywiście testujemy.
- **Wszystkie udokumentowane gwarancje muszą być prawdą.** „Leakage-safe" w README = leakage-safe w kodzie. Stąd zależność od planu remediation.

---

## 3. Blok R0 — Blockers (must-have przed 1.0)

### R0-1: Ukończenie planu leakage-safe (hard dependency)

Wszystkie zadania **P0** z [`2026-04-18_leakage_safe_remediation_plan.md`](./2026-04-18_leakage_safe_remediation_plan.md) muszą być zmergowane. Co najmniej **P1-5** (permuted target diagnostic) powinno też być, bo 1.0 będzie marketowane jako „leakage-safe library".

**DoD:**
- [ ] P0-1, P0-2, P0-3, P0-4 z planu leakage-safe — DONE.
- [ ] P1-5 (permuted target) — DONE.
- [ ] Reruny wszystkich datasetów ze świeżą metadata. Co najmniej 1 dataset z `run_grade=report-grade`.

---

### R0-2: Decyzja licencyjna

⚠ **DECYZJA WYMAGANA — najwyższy priorytet:**

Aktualna licencja `CC-BY-NC-4.0` ma 3 problemy dla 1.0 biblioteki Pythonowej:

1. **Non-Commercial:** zabrania komercyjnego użycia. Większość firm po prostu nie może użyć. PyPI nie ma kategorii „NC".
2. **CC nie jest licencją software.** CC sami odradzają (https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software). Brak klauzul o patentach, gwarancjach, distrybucji binariów.
3. **Niespójność:** klasyfikator w pyproject mówi `Topic :: Scientific/Engineering`, ale brakuje `License ::` classifier.

**Opcje (do wyboru przez leada):**

| Opcja | Plus | Minus |
|---|---|---|
| (a) **Apache-2.0** | Standard branżowy, klauzula patentowa, kompatybilna z większością | „Otwiera" kod komercyjnie — OK jeśli chcemy adopcję |
| (b) **MIT** | Najprostsza, najszersza adopcja | Brak klauzuli patentowej |
| (c) **AGPL-3.0** | Copyleft, wymusza otwartość pochodnych | Większość firm odrzuca z automatu |
| (d) **Dual: AGPL + komercyjna** | Open-core model | Wymaga obsługi prawnej |
| (e) **Zostaje CC-BY-NC-4.0** | Brak zmiany | NIE wydajemy 1.0 jako biblioteki — to wtedy „research artifact", inny komunikat |

**Sugestia juniora:** (a) Apache-2.0 — zgodne z ekosystemem ML (numpy, pandas, scikit-learn, xgboost wszystkie BSD/Apache). Paper i artykuł naukowy zostają pod CC-BY-NC.

**Akcje gdy decyzja zapadnie:**
1. Aktualizuj `LICENSE` (pełen tekst nowej licencji).
2. `pyproject.toml`: `license = "Apache-2.0"` + `classifiers += ["License :: OSI Approved :: Apache Software License"]`.
3. README badge.
4. Wpis w CHANGELOG: `### Changed - License changed from CC-BY-NC-4.0 to Apache-2.0`.
5. CITATION.cff: pole `license`.
6. Skontaktuj autorów paper, czy się zgadzają na zmianę licencji repo.

**DoD:**
- [ ] Decyzja udokumentowana w `docs/plan/license_decision.md`.
- [ ] LICENSE, pyproject, README, CITATION zaktualizowane.

---

### R0-3: Heavy deps → optional extras

**Cel:** `pip install stat-context` powinno ściągnąć tylko numpy/pandas/sklearn. Backendy ML (xgboost, lightgbm, catboost) i parquet powinny być opt-in.

**Pliki:**
- [`pyproject.toml`](../../pyproject.toml)
- [`README.md`](../../README.md) — sekcja Installation
- [`CHANGELOG.md`](../../CHANGELOG.md)

**Kroki:**

1. W `pyproject.toml` — `dependencies` zostawiamy minimalne:
   ```toml
   dependencies = [
       "numpy>=1.21.0",
       "pandas>=1.3.0,<3.0.0",
       "scikit-learn>=1.0.0,<2.0.0",
       "joblib>=1.1.0",
       "tomli>=2.0.0;python_version<'3.11'",
   ]
   ```
   Wyrzucone: `xgboost`, `lightgbm`, `catboost`, `pyarrow`, `toml` (deprecated).

2. Nowe extras:
   ```toml
   [project.optional-dependencies]
   xgboost = ["xgboost>=1.5.0,<3.0.0"]
   lightgbm = ["lightgbm>=4.0.0"]
   catboost = ["catboost>=1.2.0"]
   parquet = ["pyarrow>=10.0.0"]
   models = ["stat-context[xgboost,lightgbm,catboost]"]
   all = ["stat-context[dev,data,viz,docs,models,parquet]"]
   ```

3. **Lazy imports** w kodzie. Każdy moduł, który dotychczas importował heavy dep na top-level, musi:
   - Przenieść import do funkcji.
   - Rzucać czytelny `ImportError` z instrukcją: `"xgboost is not installed. Install it via: pip install stat-context[xgboost]"`.

   Sprawdź [`sce/models.py`](../../sce/models.py), [`sce/model_presets.py`](../../sce/model_presets.py), [`sce/io/__init__.py`](../../sce/io/__init__.py).

4. CI: dodać job `test-minimal` instalujący tylko `pip install -e .` (bez extras) i odpalający `pytest -k "not requires_xgboost and not requires_parquet"`. Markery do dodania w testach.

⚠ **DECYZJA WYMAGANA:** czy `pyarrow` ma być required (parquet to backbone naszego data flow)? Sugestia: zostaje w `[parquet]` extras, bo użytkownik biblioteki SCE niekoniecznie pracuje z parquetem — nasze `data/` jest tylko dla benchmarków.

**Testy:**
- `tests/test_imports.py` (NOWY) — `test_core_import_works_without_xgboost`: tymczasowo usuwa `xgboost` z `sys.modules`, monkeypatchem ustawia `sys.modules['xgboost'] = None`, importuje `sce`, oczekuje brak crashu.

**DoD:**
- [ ] Core install < 100MB (sprawdź `pip install --dry-run stat-context`).
- [ ] CI matrix ma job `test-minimal` zielony.
- [ ] README aktualizuje przykłady `pip install stat-context[xgboost]`.

---

### R0-4: Usunięcie deprecated API

**Cel:** w 1.0 nie ma `DeprecationWarning`. Każde `DEPRECATED` z 0.x usuwamy.

**Pliki:**
- [`sce/config.py`](../../sce/config.py) — `hierarchy`, `additional_categorical_cols`.
- [`sce/stats.py`](../../sce/stats.py) — `hierarchy=`, `additional_categorical_cols=`, `include_quantiles=` parametry w 4 funkcjach.

**Kroki:**

1. Wyszukaj `grep -rn "DeprecationWarning\|DEPRECATED" sce/`.
2. Usuń pola/parametry oraz cały kod migracyjny w `__post_init__` i wewnątrz funkcji.
3. Usuń wzmianki z docstringów.
4. Zaktualizuj wszystkie testy używające starego API (jeśli są).
5. CHANGELOG:
   ```markdown
   ### Removed (BREAKING)
   - `ContextConfig.hierarchy` (use `categorical_cols`)
   - `ContextConfig.additional_categorical_cols` (use `categorical_cols`)
   - `compute_aggregations(hierarchy=...)` parameter
   - `compute_aggregations(additional_categorical_cols=...)` parameter
   - `compute_aggregations(include_quantiles=...)` parameter (use `AggregationMethod.Q25/Q75` in `methods` list)
   ```

**Testy:**
- `test_no_deprecation_warnings` — `pytest -W error::DeprecationWarning -W error::FutureWarning`. Dodać do CI jako osobny step.

**DoD:**
- [ ] `grep -rn "DEPRECATED\|DeprecationWarning" sce/` zwraca 0 trafień.
- [ ] Cała sekcja Removed w CHANGELOG opisana.
- [ ] CI step z `-W error::DeprecationWarning` przechodzi.

---

### R0-5: Public API freeze

**Cel:** ustalić finalną listę symboli w `sce/__init__.py`. Po 1.0 — żadne dodanie/usunięcie bez wersji minor/major.

**Pliki:**
- [`sce/__init__.py`](../../sce/__init__.py)

**Kroki:**

1. Audytuj każdy obecny export ([sce/__init__.py L7-L66](../../sce/__init__.py#L7)).
2. Dla każdego symbolu odpowiedz na 3 pytania:
   - Czy użytkownik biblioteki tego potrzebuje? (Yes → zostaje)
   - Czy jest udokumentowany w README/docs? (No → albo dokumentuj, albo wywal)
   - Czy ma testy? (No → albo dopisz, albo nie eksportuj)
3. Stwórz tabelę propozycji w [`docs/plan/public_api_1_0.md`](./public_api_1_0.md) (NOWY plik) — kolumny: symbol, status (keep/remove/internal), uzasadnienie.
4. Zatwierdź z leadem przed mergem.

**Sugerowana lista PUBLIC dla 1.0** (do weryfikacji):

```python
__all__ = [
    # Core engine
    "StatisticalContextEngine",
    "ContextConfig",
    "AggregationMethod",
    "CleanupConfig",
    "detect_categorical_columns",
    # Pipeline helpers
    "create_sce_pipeline",
    "fit_context_pipeline",
    # Baseline variants (paper section)
    "SUPPORTED_CONTEXT_VARIANTS",
    "get_context_variant_label",
]
```

**Sugerowane do oznaczenia jako INTERNAL** (przeniesienie do `sce._internal` lub usunięcie z `__init__`):

- `LMFeatureSelector`, `compute_lm_statistics`, `select_significant_features` → 17% coverage, oznaczone jako experimental
- `FeatureCombinationSearch`, `SearchResult`, `SearchSummary` → tooling do experiments, nie biblioteka
- `aggregate_importance`, `run_iterative_pruning` → tooling
- `SUPPORTED_MODEL_TYPES`, `build_model`, `get_model_label`, `model_supports_gpu`, `load_model_presets`, `load_xgboost_presets`, `resolve_model_presets`, `resolve_xgboost_presets` → wewnętrzna konfiguracja experiments
- `FeatureCleanupPipeline` → ⚠ DECYZJA, przydatne ale heavy
- `resolve_context_variant_methods` → internal helper

⚠ **DECYZJA WYMAGANA:** czy w 1.0 zostawiamy public API tylko core engine (10 symboli), czy szerokie (25+)? Sugestia: **wąskie**, reszta dostępna przez `sce.experimental.*` z jawnym `_experimental_warning()`.

**DoD:**
- [ ] Tabela API w `docs/plan/public_api_1_0.md` zatwierdzona.
- [ ] `sce/__init__.py` ma finalne `__all__`.
- [ ] Dla każdego internal symbolu: usunięty z `__init__`, lub przeniesiony do `sce.experimental` submodule.
- [ ] Test: `from sce import *` daje dokładnie listę z `__all__`.

---

### R0-6: Usunięcie `print()` z biblioteki

**Cel:** Library NIE może `print()`. Tylko `logging`.

**Pliki:**
- [`sce/engine.py`](../../sce/engine.py) L120, L122 (i ewentualnie więcej — sprawdź `grep -n "print(" sce/`).
- [`sce/cli.py`](../../sce/cli.py) — `print` jest OK w CLI (osobna ścieżka).

**Kroki:**

1. `grep -rn "print(" sce/ | grep -v cli.py` — lista wszystkich.
2. Każde `print(...)` → `logger.info(...)` lub usuwamy duplikat z `logger.info` powyżej.

**Testy:**
- `tests/test_no_print_in_library.py` (NOWY) — AST scan analogiczny do P2-10 z planu leakage. Skanuje `sce/` poza `cli.py`, wyrzuca fail jeśli znajdzie wywołanie `print`.

**DoD:**
- [ ] 0 wywołań `print()` w `sce/` (poza `cli.py`).
- [ ] Test guard zielony.

---

### R0-7: Repo cleanup (wyrzucenie build artifacts z git)

**Cel:** `git ls-files` nie zwraca `build/`, `dist/`, `htmlcov/`, `catboost_info/`, `experiment_debug.log`, `*.egg-info/`, `.release-smoke/`.

**Kroki:**

1. Sprawdź `.gitignore` — dodaj brakujące:
   ```gitignore
   *.egg-info/
   build/
   dist/
   htmlcov/
   catboost_info/
   experiment_debug.log
   .release-smoke/
   ```
2. `git rm -r --cached sce.egg-info stat_context.egg-info build dist htmlcov catboost_info experiment_debug.log .release-smoke`
3. Commit: `[release-1.0][R0-7] Remove build artifacts from version control`.

**DoD:**
- [ ] `git ls-files | grep -E '(egg-info|^build/|^dist/|htmlcov)'` jest pusty.
- [ ] Następny `pip install -e .` nie commituje śmieci.

---

## 4. Blok R1 — Quality bar

### R1-8: Coverage gating

**Cel:** Min 80% coverage łącznie. Modules < 50% nie idą do public API (już zaadresowane w R0-5 — `selection.py`).

**Kroki:**

1. W `pyproject.toml` dodać:
   ```toml
   [tool.coverage.report]
   fail_under = 80
   exclude_lines = [
       "pragma: no cover",
       "if TYPE_CHECKING:",
       "raise NotImplementedError",
   ]
   ```
2. CI step: `pytest --cov=sce --cov-fail-under=80`.
3. Per-module raport: `pytest --cov=sce --cov-report=term-missing` — jeśli któryś public module < 60%, tworzymy ticket.

⚠ **DECYZJA WYMAGANA:** próg 80% globalny vs per-module. Sugestia: globalny 80%, plus per-public-module ≥ 60%.

**DoD:**
- [ ] CI fail jeśli coverage < 80%.
- [ ] Aktualne coverage ≥ 80% (jeśli nie — dopisać testy w R1).

---

### R1-9: Type checking

**Cel:** `mypy sce` przechodzi w strict mode (przynajmniej dla public API).

**Pliki:**
- [`pyproject.toml`](../../pyproject.toml)
- CI

**Kroki:**

1. W `pyproject.toml` mypy section:
   ```toml
   [tool.mypy]
   python_version = "3.10"
   strict = false  # globalny
   warn_unused_ignores = true
   warn_redundant_casts = true

   [[tool.mypy.overrides]]
   module = "sce.engine"
   strict = true

   [[tool.mypy.overrides]]
   module = "sce.config"
   strict = true
   ```
2. CI step: `mypy sce` (nie `tests/`).

⚠ **DECYZJA WYMAGANA:** strict od razu czy stopniowo? Sugestia: strict tylko dla `engine.py` i `config.py` w 1.0; reszta w 1.1.

**DoD:**
- [ ] `mypy sce` w CI zielony.
- [ ] Public API ma kompletne type hints.

---

### R1-10: Min-version compatibility test

**Cel:** Sprawdzamy, że deklarowane minima działają.

**Kroki:**

1. Nowy job CI `test-min-versions`:
   ```yaml
   - name: Install minimum versions
     run: pip install -e . numpy==1.21.0 pandas==1.3.0 scikit-learn==1.0.0
   - name: Run smoke tests
     run: pytest tests/test_engine.py tests/test_config.py
   ```

⚠ **DECYZJA WYMAGANA:** jeśli `numpy==1.21.0` ma konflikt z aktualnymi wymaganiami `pandas`/`sklearn` (możliwe), trzeba podnieść floor. Sugestia: `numpy>=1.23, pandas>=2.0, scikit-learn>=1.3` jako bezpieczny baseline 2026.

**DoD:**
- [ ] CI job `test-min-versions` zielony.
- [ ] Bumping floors udokumentowany w CHANGELOG (BREAKING).

---

### R1-11: Doctest / example smoke test

**Cel:** Każdy fragment kodu w README i docstringach `sce/` faktycznie działa.

**Kroki:**

1. CI step: `pytest --doctest-modules sce/` (przynajmniej dla `engine.py` i `config.py`).
2. Sprawdź każdy snippet w README ręcznie (`examples/basic_usage.py` powinno być 1:1 z README quickstart).
3. Dodać `examples/basic_usage.py` jako test integracyjny: `pytest tests/test_examples.py` uruchamia `examples/basic_usage.py` i sprawdza brak crash.

**DoD:**
- [ ] README quickstart = `examples/basic_usage.py`.
- [ ] CI uruchamia example bez crashu.
- [ ] Docstring examples (`>>>`) wszystkie zielone.

---

## 5. Blok R2 — Documentation pass

### R2-12: README dla 1.0

**Cel:** README to pierwsza rzecz na PyPI. Musi być perfekcyjny.

**Kroki:**

1. Aktualizuj badge’y (license po R0-2, version, python).
2. Dodaj „Stability" badge: `[![Stability: Stable](https://img.shields.io/badge/stability-stable-brightgreen.svg)]()`.
3. Sekcja **Migration from 0.x** — krótka tabela co usunięte/przemianowane.
4. Sekcja **Leakage-Safe Guarantees** — explicit list co biblioteka gwarantuje (po remediation):
   - Out-of-fold cross-fitting (default)
   - Train-only fit when `transform()` called separately
   - Hard guard for temporal datasets
5. Sekcja **Limitations** — co NIE robi (np. classification, time-series cross-fit dopiero od X).
6. Wszystkie linki absolutne (`https://github.com/...`) — relative linki łamią się na PyPI.

**DoD:**
- [ ] README rendered preview na TestPyPI wygląda poprawnie.
- [ ] Wszystkie linki działają z poziomu PyPI.

---

### R2-13: CHANGELOG dla 1.0.0

**Cel:** Pełna sekcja `[1.0.0] - YYYY-MM-DD` z kategoriami: `Changed (BREAKING)`, `Removed (BREAKING)`, `Added`, `Fixed`, `Security`.

**Template do użycia:**

```markdown
## [1.0.0] - 2026-MM-DD

First stable release. **Breaking changes vs 0.x — see Migration section in README.**

### Changed (BREAKING)

- License changed from CC-BY-NC-4.0 to <new license>
- Heavy ML backends (xgboost, lightgbm, catboost) moved to optional extras
- Minimum supported versions: numpy>=X, pandas>=Y, scikit-learn>=Z
- Public API frozen — see migration guide

### Removed (BREAKING)

- `ContextConfig.hierarchy` (use `categorical_cols`)
- `ContextConfig.additional_categorical_cols` (use `categorical_cols`)
- `compute_aggregations(hierarchy=)` parameter
- `compute_aggregations(additional_categorical_cols=)` parameter
- `compute_aggregations(include_quantiles=)` parameter
- `LMFeatureSelector`, `FeatureCombinationSearch`, ... moved to `sce.experimental`

### Added

- `cross_fit_strategy` parameter (random | time | off) in ContextConfig
- Hard guard: temporal split + random cross-fit raises ValueError
- Train-only categorical encoding and feature pruning
- Permuted-target diagnostic (`scripts/diagnostics/permuted_target.py`)
- Shuffled-groups diagnostic
- Cross-fit A/B diagnostic
- Feature dominance audit
- Run metadata: git_sha, config_hash, run_grade, full SCE flag set
- `--run-grade` CLI flag (exploratory | diagnostic | report-grade)

### Fixed

- Removed all `print()` calls from library code (use logging)
- Cleaned up build artifacts from version control
```

**DoD:**
- [ ] CHANGELOG zaktualizowany.
- [ ] [Unreleased] sekcja pusta po release.

---

### R2-14: Migration guide

**Cel:** `docs/migration/0.x_to_1.0.md` — krok po kroku jak użytkownicy 0.x mają zmigrować.

**Kroki:**

1. Stwórz plik z sekcjami:
   - Installation changes (`pip install stat-context[xgboost]` zamiast samo `stat-context`)
   - Removed parameters → replacement
   - Renamed/moved symbols
   - Behavior changes (jeśli są — np. nowy domyślny `cross_fit_strategy`)
2. Dodaj link do README.

**DoD:**
- [ ] Plik istnieje, każda zmiana ma sekcję „przed / po".

---

### R2-15: API reference rebuild

**Kroki:**

1. `mkdocs build --strict` musi przejść bez ostrzeżeń.
2. Wszystkie public symbols z [§R0-5](#r0-5-public-api-freeze) mają sekcję w `docs/api/`.
3. Deploy GitHub Pages — workflow `docs.yml`.

**DoD:**
- [ ] `mkdocs build --strict` zielony.
- [ ] https://joint-hubs.github.io/sce pokazuje 1.0 docs.

---

## 6. Blok R3 — Release infra

### R3-16: SECURITY.md

**Plik:** `SECURITY.md` (NOWY).

**Treść:**

```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |
| 0.x     | :x: (security fixes only until 2026-12-31) |

## Reporting a Vulnerability

Please report vulnerabilities privately via GitHub Security Advisories:
https://github.com/joint-hubs/sce/security/advisories/new

We aim to acknowledge within 5 business days and patch within 30 days for critical issues.
```

**DoD:**
- [ ] `SECURITY.md` istnieje, link w README.

---

### R3-17: Release smoke test workflow

**Cel:** end-to-end test działającej wheel'i przed publikacją na PyPI.

**Pliki:**
- [`.github/workflows/release.yml`](../../.github/workflows/release.yml)

**Kroki:**

1. W jobie `test` (po build, przed `publish-testpypi`) dodać:
   ```yaml
   - name: Run smoke pipeline
     run: |
       pip install dist/*.whl[xgboost]
       python -c "
       import pandas as pd
       from sce import StatisticalContextEngine, ContextConfig
       df = pd.DataFrame({'city': ['a','a','b','b','c','c'] * 20,
                          'price': [10,12,20,22,30,33] * 20})
       cfg = ContextConfig(target_col='price', categorical_cols=['city'],
                           use_cross_fitting=True, n_folds=2, min_group_size=2)
       enriched = StatisticalContextEngine(cfg).fit_transform(df)
       assert len(enriched.columns) > len(df.columns), 'No features added'
       print(f'Smoke OK: {len(enriched.columns)} columns')
       "
   ```

**DoD:**
- [ ] Smoke step zielony w CI.

---

### R3-18: TestPyPI dry-run

**Cel:** opublikować `1.0.0rc1` na TestPyPI, zainstalować w czystym venv, uruchomić smoke. **DOPIERO POTEM** tag `v1.0.0`.

**Procedura (lead wykonuje):**

1. Bump version na `1.0.0rc1` w `sce/__init__.py` i `pyproject.toml`.
2. Tag `v1.0.0rc1` → push → GH Action automatycznie publikuje na TestPyPI.
3. Czysty venv: `python -m venv /tmp/sce-test && /tmp/sce-test/bin/pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ stat-context==1.0.0rc1`.
4. Uruchom `examples/basic_usage.py`.
5. Jeśli OK → bump na `1.0.0`, tag `v1.0.0`, push.

⚠ **DECYZJA WYMAGANA:** czy chcemy pełny `rc1` cykl (1-2 tygodnie publicznego testingu), czy szybki sanity check + production tag? Sugestia: rc1, 1 tydzień minimum.

**DoD:**
- [ ] `1.0.0rc1` na TestPyPI.
- [ ] Smoke zainstalowanej wheels OK.
- [ ] Brak issue zgłoszonych w 7 dniach.

---

### R3-19: Version bump automation

**Pliki:**
- [`scripts/bump_version.py`](../../scripts/bump_version.py) — sprawdzić, czy działa.

**Kroki:**

1. Zweryfikuj, że `bump_version.py` aktualizuje wszystkie miejsca: `sce/__init__.py`, `pyproject.toml`, `CITATION.cff`.
2. Test ręczny: `python scripts/bump_version.py 1.0.0rc1 --dry-run`.

**DoD:**
- [ ] Skrypt działa, jest udokumentowany w CONTRIBUTING.

---

## 7. Blok R4 — Optional polish (nice-to-have, nie blokuje 1.0)

### R4-20: SBOM generation

`pip install cyclonedx-bom` + `cyclonedx-py environment > sbom.json`. Załączyć do GitHub release jako asset.

### R4-21: Sigstore / attestations

PyPI trusted publishing OIDC już generuje attestations. Sprawdź, czy są widoczne na PyPI dla 1.0.0rc1.

### R4-22: Reproducible builds

`SOURCE_DATE_EPOCH` w build action.

### R4-23: Performance benchmarks

`asv` (airspeed velocity) — track regression w czasie. Out-of-scope dla 1.0, ale wspomnieć w roadmap.

### R4-24: Python 3.13 support

Dodać do CI matrix gdy jest stabilne.

---

## 8. Decyzje WYMAGANE od leada (zbiorczo)

| # | Ticket | Decyzja | Sugestia juniora |
|---|---|---|---|
| RD1 | R0-2 | **Licencja** (Apache-2.0 vs MIT vs AGPL vs zostaje CC-BY-NC vs dual) | Apache-2.0 |
| RD2 | R0-3 | Czy `pyarrow` ma być required czy w `[parquet]` extras? | extras |
| RD3 | R0-5 | Wąskie public API (~10 symboli) vs szerokie (~25, jak teraz)? | wąskie + `sce.experimental` |
| RD4 | R1-8 | Coverage threshold globalny vs per-module? | 80% global + 60% per public module |
| RD5 | R1-9 | mypy strict od razu vs stopniowo? | strict tylko engine+config w 1.0 |
| RD6 | R1-10 | Bumping deps minimum (3.9 EOL) — pozwalamy na 3.9 czy minimum 3.10? | min 3.10 (3.9 EOL Oct 2025) |
| RD7 | R3-18 | Pełen rc cycle (1+ tydzień) vs szybki sanity? | rc1 minimum 1 tydzień |
| RD8 | (cross) | Czy `1.0.0` od razu czy `0.4.0` najpierw (wszystkie zmiany leakage-safe + breaking, ale jeszcze ostatni 0.x)? | 1.0.0 — jeśli i tak są BREAKING, semver wymusza major |
| RD9 | (cross) | Co z paczką PyPI `stat-context` vs `sce`? Rezerwujemy obie czy zostawiamy `stat-context`? | Zostaw `stat-context` (już zarejestrowane), w README mocno podkreśl |

---

## 9. Tracking checklist

Po każdym mergu wpisz nr PR.

### R0 — Blockers
- [ ] **R0-1** Leakage-safe P0+P1-5 done (zewnętrzny plan)
- [ ] **R0-2** Licencja zmieniona (PR #___)
- [ ] **R0-3** Heavy deps → optional extras (PR #___)
- [ ] **R0-4** Deprecated API removed (PR #___)
- [ ] **R0-5** Public API freeze (PR #___)
- [ ] **R0-6** print() removed from library (PR #___)
- [ ] **R0-7** Repo cleanup (PR #___)

### R1 — Quality
- [ ] **R1-8** Coverage gating ≥80% (PR #___)
- [ ] **R1-9** mypy strict for engine+config (PR #___)
- [ ] **R1-10** Min-version CI job (PR #___)
- [ ] **R1-11** Doctest + examples in CI (PR #___)

### R2 — Docs
- [ ] **R2-12** README 1.0 (PR #___)
- [ ] **R2-13** CHANGELOG 1.0.0 (PR #___)
- [ ] **R2-14** Migration guide (PR #___)
- [ ] **R2-15** API reference rebuild (PR #___)

### R3 — Release infra
- [ ] **R3-16** SECURITY.md (PR #___)
- [ ] **R3-17** Release smoke test (PR #___)
- [ ] **R3-18** TestPyPI rc1 → 1 week soak → 1.0.0
- [ ] **R3-19** bump_version.py weryfikacja (PR #___)

### R4 — Optional
- [ ] R4-20 SBOM
- [ ] R4-21 Attestations
- [ ] R4-22 Reproducible builds
- [ ] R4-23 Benchmarks
- [ ] R4-24 Python 3.13

---

## 10. Sekwencja merge'y rekomendowana

1. **Pre-req:** ukończ leakage-safe P0+P1-5 (zewnętrzny plan).
2. **R0-2** (licencja) — najwcześniej, bo wpływa na wszystko poniżej.
3. **R0-7** (cleanup) — niezależny, łatwy.
4. **R0-6** (print removal) — niezależny, łatwy.
5. **R0-3** (deps) + **R0-4** (deprecated) + **R0-5** (API freeze) — w tej kolejności, bo R0-5 finalizuje to co R0-4 usunęło.
6. **R1-9** (mypy) — wymaga zamknięcia API.
7. **R1-8** (coverage) + **R1-10** (min-versions) + **R1-11** (doctest) — równolegle.
8. **R2-12, R2-13, R2-14, R2-15** — równolegle, ostatni etap przed releasem.
9. **R3-16, R3-17, R3-19** — przygotowanie infra.
10. **R3-18** — `1.0.0rc1` na TestPyPI, 1 tydzień soak, potem `1.0.0` na PyPI.

---

## 11. Pre-flight checklist (1 dzień przed tagiem `v1.0.0`)

Lead odhaczyć przed pushem tagu:

- [ ] Wszystkie P0/P1 z leakage-safe — zmergowane, CI zielone.
- [ ] Wszystkie R0/R1/R2/R3 ticketu — zmergowane.
- [ ] `git status` clean, branch `main` aktualny.
- [ ] `pytest` zielony lokalnie (full suite + doctest).
- [ ] `mypy sce` zielony.
- [ ] `ruff check sce tests` zielony.
- [ ] `mkdocs build --strict` zielony.
- [ ] `python -m build && twine check dist/*` zielony.
- [ ] `1.0.0rc1` na TestPyPI od ≥7 dni, brak otwartych issue.
- [ ] CHANGELOG: data wstawiona, [Unreleased] pusty.
- [ ] CITATION.cff: nowa wersja, data.
- [ ] README badge "stable" widoczny.
- [ ] LICENSE = nowa licencja (R0-2).
- [ ] Wszystkie linki w README absolute, działają na TestPyPI rendered preview.
- [ ] Backup poprzedniego release’u (`stat-context==0.3.5`) zachowany — nie usuwać z PyPI.

---

## 12. Out of scope dla 1.0

- Classification support (regression-only zostaje).
- Time-series specific features (poza temporal cross-fit z R0-1).
- GPU acceleration core (XGBoost backend ma własne).
- Distributed processing (Dask/Ray).
- New aggregation methods.
- Sigstore signing core artifacts (R4-21 to attestations only).

Te trafiają do `roadmap.md` (osobny plik, R3+).

---

## 13. Co NIE jest w tym dokumencie i kogo o to zapytać

- Decyzje paper-related (np. czy 1.0 wymaga zaktualizowanego paper) → autorzy / `auditor` agent.
- PyPI org / trusted publishing setup → tech lead (musi mieć dostęp do PyPI).
- Marketing / blog post / announcement → poza scope, lead decyduje.
- DOI dla 1.0.0 (Zenodo) → opcjonalne, lead decyduje.
