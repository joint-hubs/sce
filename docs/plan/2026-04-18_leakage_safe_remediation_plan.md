# Leakage-Safe Remediation Plan (junior-engineer playbook)

**Created:** 2026-04-18
**Owner:** junior dev (implementation), tech lead (sign-off)
**Source audit:** rozmowa 2026-04-18 — checklista „Leakage-Safe Checklist" zastosowana do bieżącego kodu.
**Reference docs:** `.github/skills/sce-algorithm/SKILL.md`, `paper_overleaf_format.txt`

---

## 0. Jak korzystać z tego dokumentu

1. Zadania są ułożone w blokach **P0 → P1 → P2**. Bloki wyższego priorytetu muszą być zmergowane przed niższymi.
2. Każde zadanie ma identyczną strukturę:
   - **Cel** — co i dlaczego.
   - **Pliki** — gdzie modyfikujemy.
   - **Kroki** — co dokładnie robimy (możliwie atomowe commity).
   - **Testy** — co musi być zielone.
   - **Decyzje wymagane** — co MUSISZ skonsultować z leadem zanim zaczniesz.
   - **Definition of Done (DoD)** — kryterium zamknięcia ticketu.
3. **Nie improwizuj** w punktach oznaczonych `⚠ DECYZJA WYMAGANA`. Zatrzymaj się i zapytaj.
4. Każde zadanie ⇒ osobny PR. Tytuł PR wg konwencji: `[leakage-safe][P{0|1|2}-{nr}] krótki opis`.
5. Po skończeniu każdego zadania zaktualizuj checklistę w sekcji [§7](#7-tracking-checklist) na końcu dokumentu.

---

## 1. Słownik

| Pojęcie | Definicja w naszym repo |
|---|---|
| **Run grade** | Etykieta na runie: `exploratory` (szybki, brudny), `diagnostic` (audytowy, np. permuted target), `report-grade` (idzie do paper/raportu). |
| **OOF / cross-fit** | Out-of-fold aggregation w `StatisticalContextEngine._fit_transform_cross_fitted`. |
| **Temporal cross-fit** | Cross-fit, w którym fold walidacyjny zawiera obserwacje **późniejsze** niż foldy używane do liczenia statystyk (`TimeSeriesSplit`-podobnie). |
| **Config hash** | `sha256(canonical_toml_bytes).hexdigest()[:16]` — deterministyczny, krótki ID configu. |
| **Run ID** | `f"{config_name}__{git_sha[:8]}__{config_hash[:8]}__{timestamp}"`. |
| **Train-only fit** | Każdy preprocessing krok ma fitowane parametry **wyłącznie** na rowach treningowych po splicie. |

---

## 2. Mapa modułów i co zmieniamy w każdym

| Moduł | Co tu robimy w tym planie |
|---|---|
| [`sce/config.py`](../../sce/config.py) | nowe pole `cross_fit_strategy`; walidacja temporal+random |
| [`sce/engine.py`](../../sce/engine.py) | `random_state` z configu, `cross_fit_strategy`, opcjonalnie `time_col` w cross-fitcie |
| [`scripts/run.py`](../../scripts/run.py) | metadata (SHA, hash, seed, flagi), train-only encoding/pruning, hard-guard temporal, run_grade flag |
| `scripts/diagnostics/` (NOWY folder) | `permuted_target.py`, `shuffled_groups.py`, `crossfit_ab.py`, `feature_dominance.py` |
| [`tests/`](../../tests/) | nowe testy: `test_leakage_guards.py`, `test_metadata.py`, `test_diagnostics.py` |
| [`docs/plan/`](.) | ten plik + raport postępu |

---

## 3. Blok P0 — Hard fixes (must-have przed jakimkolwiek report-grade runem)

### P0-1: Hard guard: temporal split + random cross-fit zabronione

**Cel:** zlikwidować [Hard Fail Rule 2](../experiments.md) dla configów `rossmann_daily.toml` i `walmart_weekly.toml`. Aktualnie `KFold(shuffle=True, random_state=42)` w [sce/engine.py L329](../../sce/engine.py) miesza foldy losowo, więc statystyki dla obserwacji z roku N mogą pochodzić z roku N+1 → temporal leakage.

**Pliki:**
- [`sce/config.py`](../../sce/config.py) — dodać pole.
- [`sce/engine.py`](../../sce/engine.py) — użyć pola.
- [`scripts/run.py`](../../scripts/run.py) — wymusić zgodność splitu i cross-fitu.
- [`tests/test_leakage_guards.py`](../../tests/test_leakage_guards.py) — NOWY.

**Kroki:**

1. W `ContextConfig` dodać:
   ```python
   from typing import Literal
   cross_fit_strategy: Literal["random", "time", "off"] = "random"
   time_col: Optional[str] = None  # wymagane gdy cross_fit_strategy == "time"
   ```
   W `__post_init__` walidacja:
   ```python
   if self.cross_fit_strategy == "time" and not self.time_col:
       raise ValueError("cross_fit_strategy='time' requires time_col to be set")
   if self.cross_fit_strategy == "off" and self.use_cross_fitting:
       raise ValueError("cross_fit_strategy='off' is incompatible with use_cross_fitting=True")
   ```

2. W `StatisticalContextEngine._fit_transform_cross_fitted` ([sce/engine.py L301](../../sce/engine.py)):
   - Jeśli `self.config.cross_fit_strategy == "random"`: zostaw `KFold` ALE użyj `random_state=getattr(self.config, "random_state", 42)` (patrz P2-9).
  - Jeśli `"time"`:
     ```python
     from sklearn.model_selection import TimeSeriesSplit
     X_sorted = X_reset.sort_values(self.config.time_col, kind="mergesort")
     kf = TimeSeriesSplit(n_splits=self.config.n_folds)
     fold_indices = list(kf.split(X_sorted))
     # uwaga: po sortowaniu indeksy iloc są względem X_sorted, nie X_reset
     ```
     `TimeSeriesSplit` zwraca rosnące przedziały — fold m używa wszystkiego < t_m do liczenia statystyk, fold m+1 nie wraca do przeszłości.

   ⚠ **DECYZJA WYMAGANA:**
   `TimeSeriesSplit` ma asymetryczny rozkład danych (pierwszy fold „test" jest mały, ostatni duży). Czy chcemy:
   - (a) klasyczny `TimeSeriesSplit` (sklearn), albo
   - (b) **rolling window** (każdy fold tej samej długości)?
   Nie zaczynaj implementacji wariantu (b) bez decyzji leada.

3. W [`scripts/run.py::_build_sce_config`](../../scripts/run.py) zastąpić bieżący `logger.warning(...)` (linie ~256-264) twardym guardem:
   ```python
   split_strategy = config.get("split", {}).get("strategy", "random")
   sce_use_cf = sce_cfg.get("use_cross_fitting", True)
   sce_cf_strategy = sce_cfg.get("cross_fit_strategy", "random")
   if split_strategy == "temporal" and sce_use_cf and sce_cf_strategy == "random":
       raise ValueError(
           "Temporal split forbids random cross-fit (causes temporal leakage). "
           "Set sce.cross_fit_strategy='rolling' (recommended) or sce.use_cross_fitting=false."
       )
   ```
   Przekaż `cross_fit_strategy` i `time_col` (z `split.time_col`) do `ContextConfig(...)`.

4. Zaktualizować configi temporal:
  - [`configs/rossmann_daily.toml`](../../configs/rossmann_daily.toml): w sekcji `[sce]` dodać `cross_fit_strategy = "rolling"`.
  - [`configs/walmart_weekly.toml`](../../configs/walmart_weekly.toml): jw.
   - `m5_store_dept_daily.toml` ma już `use_cross_fitting=false` — nic nie zmieniaj, ale dodaj komentarz.

**Testy (`tests/test_leakage_guards.py`):**
- `test_temporal_with_random_crossfit_raises` — config dict z `split.strategy=temporal` + `sce.use_cross_fitting=true` + `sce.cross_fit_strategy=random` → `_build_sce_config` rzuca `ValueError`.
- `test_temporal_with_rolling_crossfit_passes` — analogiczny config z `cross_fit_strategy="rolling"` przechodzi i `ContextConfig.cross_fit_strategy == "rolling"`.
- `test_rolling_crossfit_is_monotonic` — fitujemy engine z `cross_fit_strategy="rolling"` na syntetycznych danych z `date` 2020-01-01..2020-12-31; sprawdzamy, że dla każdego foldu max(timestamp obserwacji walidacyjnych) > max(timestamp obserwacji użytych do statystyk). Jak nie umiesz tego zaaserować bez dotykania prywatnego API — dodaj tymczasowy hak `engine._last_fold_timestamps` widoczny tylko z testu (lub przekaż listę przez parametr debug).

**DoD:**
- [ ] Pole `cross_fit_strategy` istnieje, dokumentowane w docstring `ContextConfig`.
- [ ] Hard guard działa, oba configi temporal mają `cross_fit_strategy="rolling"`.
- [ ] Wszystkie 3 testy zielone.
- [ ] `python scripts/run.py --dataset rossmann_daily` działa bez warningów o temporal+random.

---

### P0-2: Train-only categorical encoding

**Cel:** `prepare_features` w [scripts/run.py L632-637](../../scripts/run.py) wywołuje `pd.Categorical(X[col]).codes` osobno na train i test. Te same wartości stringowe dostają **różne kody** w zależności od porządku/zawartości partycji → baseline porównuje jabłka z gruszkami i Rule 3 jest złamany.

**Pliki:**
- [`scripts/run.py`](../../scripts/run.py) — refaktor `prepare_features`.

**Kroki:**

1. Wydzielić nowy helper `_fit_categorical_encoder(train_df, categorical_cols) -> dict[str, pd.CategoricalDtype]`. Zwraca mapowanie kolumna → typ kategorialny utrwalony z train.
2. Wprowadzić `prepare_features(df, config, target_col, encoder=None)`:
   - Jeśli `encoder is None` → fit i zwróć go obok `(X, y, encoder)`.
   - Jeśli `encoder` przekazany → tylko transform (`pd.Categorical(df[col], categories=encoder[col].categories).codes`); nieznane kategorie dostają `-1` (kod sentinel).
3. Zmienić wszystkie call-sites w `run_experiment` i `run_search_experiment`:
   ```python
   X_train_base, y_train, cat_encoder = prepare_features(train_df, config, target_col)
   X_test_base, y_test, _ = prepare_features(test_df, config, target_col, encoder=cat_encoder)
   ```
4. `prepare_features` musi zwracać tę samą sygnaturę zawsze — zwróć `encoder=None` w trybie test-only.

⚠ **DECYZJA WYMAGANA:** sentinel `-1` dla unseen categories vs. `NaN` + downstream dropna. XGBoost akceptuje NaN. Sugestia: NaN, ale potwierdź z leadem (zachowanie historyczne to integer codes).

✅ **DOPRECYZOWANIE / STATUS (2026-04-18):**
- Decyzja D2 została wdrożona jako `NaN` dla unseen kategorii (bez sztucznego rekordu i bez losowego "unknown").
- Implementacja kataloguje unseen per kolumna (`unseen_count`, `unseen_rate`, próbki wartości) w `PreparedFeatures.unseen_categorical`.

**Testy:**
- `tests/test_run.py::test_categorical_encoding_train_test_consistent` — manualnie utworzyć train (`['a','b','c']`) i test (`['c','a','b']`), zaasertować że encoder daje te same kody dla tych samych stringów.
- `tests/test_run.py::test_unseen_category_in_test` — test ma kategorię nieobecną w train; zaasertować że nie crashuje, mapuje unseen do `NaN` i raportuje licznik unseen.

**DoD:**
- [x] `prepare_features` ma jeden source-of-truth dla kodowania.
- [x] 2 nowe testy zielone.
- [x] Reruny baseline na `rental_poland_short` dają deterministyczny wynik niezależny od permutacji wierszy testu (smoke test ręczny).

---

### P0-3: Train-only feature pruning (missing rate, zero variance)

**Cel:** [scripts/run.py L585-605](../../scripts/run.py) liczy `missing_rate` i `nunique` na **każdym** datasecie wchodzącym do `prepare_features`. Gdy aplikujemy go do test, drop-list może się różnić od train → niezgodność kolumn lub leakage informacji o teście.

**Kroki:**

1. Wydzielić `_compute_pruning_droplist(df, feature_cols, missing_threshold, drop_zero_variance) -> list[tuple[col, reason, value]]`.
2. W `prepare_features(..., droplist=None)`:
   - Jeśli `droplist is None` (tryb train) → policz droplist, zapamiętaj.
   - Jeśli `droplist` podany (tryb test) → zaaplikuj go bez liczenia własnego.
3. Zwracać `droplist` razem z encoderem (rozszerzyć tuple do `(X, y, encoder, droplist)` lub stworzyć dataclass `PreparedFeatures`).

**Sugestia:** zrób `@dataclass class PreparedFeatures(X, y, encoder, droplist)` — czytelniej niż 4-tuple.

**Testy:**
- `test_pruning_droplist_train_only` — train ma kolumnę z 50% NaN, test ma 0% NaN. Z `missing_threshold=0.4` train dropuje kolumnę; test musi też ją zdropować mimo że spełnia próg.
- `test_pruning_zero_variance_train_only` — kolumna ma 1 unikalną wartość w train, 5 w test. Z `drop_zero_variance=True` train dropuje, test dropuje analogicznie.

**DoD:**
- [ ] Pruning fitowany tylko na train.
- [ ] 2 testy zielone.

---

### P0-4: Kompletna metadata runa (SHA, config hash, seed, flagi)

**Cel:** Hard Fail „nie da się odtworzyć runa z configu i SHA". Aktualnie `metadata.json` jest minimalny.

**Pliki:**
- [`scripts/run.py`](../../scripts/run.py) — nowy helper `_collect_run_metadata`.
- `tests/test_metadata.py` — NOWY.

**Kroki:**

1. Dodaj helper:
   ```python
   import hashlib, subprocess

   def _git_sha() -> str:
       try:
           return subprocess.check_output(
               ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
           ).strip()
       except Exception:
           return "unknown"

   def _config_hash(config_path: Path) -> str:
       raw = config_path.read_bytes()
       return hashlib.sha256(raw).hexdigest()[:16]

   def _git_dirty() -> bool:
       try:
           out = subprocess.check_output(
               ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
           )
           return bool(out.strip())
       except Exception:
           return True
   ```

2. `_collect_run_metadata(config_path, config, run_grade, sce_config, model_type, model_params, runtime_seconds, metrics) -> dict` zwraca słownik z polami z [§5](#5-required-metadata-schema-pełna).

3. Wywołać go w `run_experiment` i `run_search_experiment`, zapisać do `metadata.json` (search) oraz do `experiment_results.json` (single — może obok `ExperimentResult`).

4. Dodać CLI flagę `--run-grade {exploratory,diagnostic,report-grade}` (default `exploratory`). Propagować do metadata.

5. Hard fail przy `report-grade` jeśli `_git_dirty() == True` lub `git_sha == "unknown"`. Wyjątek można przeskoczyć flagą `--allow-dirty` (do dev only, log warning).

⚠ **DECYZJA WYMAGANA:** czy `report-grade` ma BLOKOWAĆ run gdy brak diagnostyk (P1-5/6) jeszcze nie istnieje? Sugestia: **tak**, ale pierwszy merge tego ticketu może mieć łagodniejsze zachowanie (warning), a hard block włączamy w P1-8.

✅ **DOPRECYZOWANIE / STATUS (2026-04-18):**
- Decyzja D3 została wdrożona jako twardy blok: `report-grade` rzuca `RuntimeError`, gdy brakuje któregokolwiek wymagania diagnostycznego.
- Pominięte diagnostyki są raportowane w artefaktach jako wpis do `results/report_grade_blocks.jsonl` (`blocked_by`: `missing_diagnostic:*`).

**Schema metadata** — patrz [§5](#5-required-metadata-schema-pełna).

**Testy (`tests/test_metadata.py`):**
- `test_config_hash_deterministic` — ten sam plik → ten sam hash.
- `test_config_hash_changes_with_content` — zmiana 1 znaku → inny hash.
- `test_metadata_schema_complete` — uruchom `_collect_run_metadata` z mockowymi argumentami, zaasertować obecność każdego klucza ze schematu.
- `test_run_grade_dirty_repo_blocks_report_grade` — mock `_git_dirty()` → `True`, `run_grade="report-grade"` → `RuntimeError`.

**DoD:**
- [ ] `metadata.json` zawiera wszystkie pola z §5.
- [ ] CLI ma `--run-grade`.
- [ ] 4 testy zielone.

---

## 4. Blok P1 — Diagnostyki (warunek konieczny dla report-grade)

### P1-5: Permuted Target Diagnostic

**Cel:** sanity check leakage — po permutacji `y` przed splitem, SCE nie powinien dawać znaczącej przewagi. Jeśli daje, to znaczy, że feature pipeline „pamięta" target inną drogą.

**Pliki:**
- `scripts/diagnostics/__init__.py` — NOWY.
- `scripts/diagnostics/permuted_target.py` — NOWY.
- `tests/test_diagnostics.py` — NOWY.

**Kroki:**

1. Stworzyć moduł:
   ```python
   def run_permuted_target(config_name: str, n_permutations: int = 5, seed: int = 42) -> dict:
       """Returns {'baseline_rmse_real', 'sce_rmse_real',
                    'baseline_rmse_permuted_mean', 'sce_rmse_permuted_mean',
                    'sce_advantage_real', 'sce_advantage_permuted_mean',
                    'permuted_advantages': [...], 'pass': bool}.
       Pass criterion: sce_advantage_permuted_mean < 1pp (parametryzowalne)."""
   ```
2. W środku: `n_permutations` razy permutuj `df[target_col]` przed `_split_dataset` (nowy seed za każdym razem), odpal `run_experiment`-like pipeline, zbierz RMSE.
3. Zapisuj do `results/diagnostics/{config_name}/permuted_target_{timestamp}.json`.
4. CLI: `python scripts/diagnostics/permuted_target.py --dataset <name> [--n-permutations 5]`.

⚠ **DECYZJA WYMAGANA:**
- (a) Czy chcemy uruchamiać diagnostyki na **subsamplu** datasetu (np. 20k wierszy) dla szybkości? Domyślnie sugerujemy: **tak, 20k**, ale dla `report-grade` runu pełen dataset.
- (b) Próg pass/fail: 1pp przewagi RMSE czy 0.5pp? Sugestia 1pp, do ustalenia.

✅ **DOPRECYZOWANIE / STATUS (2026-04-18):**
- D4 wdrożone: diagnostyki (`permuted_target`, `shuffled_groups`, `crossfit_ab`) domyślnie używają subsamplu (`--max-rows 20000`).
- Tryb `report-grade` wymusza pełny zbiór (ignoruje subsample), a CLI wspiera też jawny `--full`.

**Testy:**
- `test_permuted_target_smoke` — odpalić na minidatasecie (`tests/conftest.py` ma fixturkę), zaasertować że zwraca dict z wymaganymi kluczami.
- `test_permuted_target_passes_on_synthetic_clean` — syntetyczny dataset z czystym SCE bez leakage; po permutacji `sce_advantage_permuted_mean ~ 0`. Może być flaky → użyć `seed` i tolerance `< 2pp`.

**DoD:**
- [x] CLI działa, generuje JSON.
- [x] 2 testy zielone.

---

### P1-6: Shuffled Group Structure Diagnostic

**Cel:** zniszczyć sensowną strukturę grup (permutacja wartości w `categorical_cols` zachowując marginalny rozkład). Jeśli SCE nadal poprawia, to znaczy że wzbogacenie nie pochodzi z prawdziwej struktury hierarchicznej.

**Pliki:**
- `scripts/diagnostics/shuffled_groups.py` — NOWY.

**Kroki:**

1. Funkcja `run_shuffled_groups(config_name, n_permutations=5, seed=42, columns=None) -> dict`.
2. Dla każdej iteracji:
   - Załaduj dataset, dla każdej kolumny w `columns` (default = `config.features.categorical`) zrób `df[col] = df[col].sample(frac=1, random_state=seed_i).values`.
   - Odpal pipeline.
   - Zbierz RMSE.
3. Pass criterion: `sce_advantage_real - sce_advantage_shuffled_mean > 50% real_advantage` (czyli przynajmniej połowa zysku znika).

⚠ **DECYZJA WYMAGANA:** czy permutować **wszystkie** kategoryczne na raz, czy każdą osobno (per-column ablation)? Sugestia: oba tryby, dwa CLI subkomandy (`--mode all` / `--mode per-column`).

✅ **DOPRECYZOWANIE / STATUS (2026-04-18):**
- D6 wdrożone: `shuffled_groups.py` wspiera oba tryby (`--mode all`, `--mode per-column`).
- Tryb `per-column` zwraca osobny breakdown per kolumna (`per_column.<col>.mean_advantage`, lista `advantages`) dla analizy, która kolumna niesie sygnał.

**Testy:** analogicznie do P1-5.

**DoD:**
- [ ] CLI działa, JSON output.
- [ ] 2 testy zielone (smoke + synthetic).

---

### P1-7: Cross-Fit A/B Diagnostic

**Cel:** automatyczne porównanie `use_cross_fitting=true` vs `false` żeby udokumentować, że cross-fit faktycznie redukuje leakage.

**Pliki:**
- `scripts/diagnostics/crossfit_ab.py` — NOWY.

**Kroki:**

1. Funkcja `run_crossfit_ab(config_name) -> dict`:
   - Run #1: pełen `run_experiment` z `use_cross_fitting=True` (override w configu in-memory).
   - Run #2: jw. z `use_cross_fitting=False`.
   - Output: `{rmse_cf, rmse_no_cf, r2_cf, r2_no_cf, leakage_signal_pp}` gdzie `leakage_signal_pp = (rmse_no_cf - rmse_cf) / rmse_no_cf * 100` ujemne zachowuje się dziwnie (bez cross-fit jest LEPIEJ na test = potencjalny leakage z train do test przez statystyki).
2. Zapis do `results/diagnostics/{config_name}/crossfit_ab_{timestamp}.json`.

**Testy:** smoke test.

**DoD:**
- [ ] CLI działa, JSON.
- [ ] 1 smoke test zielony.

---

### P1-8: Feature Dominance Audit

**Cel:** wykryć runy, w których cały zysk RMSE pochodzi z 1-3 feature'ów (sygnał, że to artefakt, nie statystyczna struktura).

**Pliki:**
- `scripts/diagnostics/feature_dominance.py` — NOWY.
- modyfikacja [`scripts/run.py`](../../scripts/run.py) — wpinanie do metadata.

**Kroki:**

1. Funkcja `audit_feature_dominance(importance_csv: Path, top_k=3, threshold_pct=70) -> dict`:
   - Wczytaj `aggregated_feature_importance.csv` z [scripts/run.py L893-895](../../scripts/run.py).
   - Posortuj malejąco.
   - Zwróć `{top_k_share_pct, top_features, dominated: bool}` gdzie `dominated = top_k_share_pct > threshold_pct`.
2. W `run_search_experiment` po wygenerowaniu importance CSV → wywołaj audyt → dopisz do `metadata.json` pole `feature_dominance`.
3. Hard fail dla `report-grade`: jeśli `dominated=True` i pierwsza dominująca feature to `<target_col>_*` z global level, blokuj promocję.

⚠ **DECYZJA WYMAGANA:** Próg `top_k=3 > 70%` — arbitralny. Sugeruję traktować jako warning na początku, z czasem podnosić.

✅ **DOPRECYZOWANIE / STATUS (2026-04-18):**
- D7 wdrożone: `audit_feature_dominance(...)` używa progu startowego `top_k=3`, `threshold_pct=70`.
- Dominacja jest sygnalizowana warningiem (z udziałem top-k i listą cech), a twardy blok `report-grade` pozostaje tylko dla przypadku `feature_dominance:target_global`.

**Testy:**
- `test_feature_dominance_synthetic` — sztuczny CSV z 1 feature mającym 90% importance → `dominated=True`.
- `test_feature_dominance_uniform` — 10 features po 10% → `dominated=False`.

**DoD:**
- [ ] Audit wpięty do search pipeline.
- [ ] `metadata.json` zawiera `feature_dominance`.
- [ ] 2 testy zielone.

---

## 5. Required metadata schema (pełna)

`metadata.json` (i sekcja `metadata` w `experiment_results.json` dla single runów) MUSI zawierać:

```json
{
  "run_id": "rental_poland_short__a1b2c3d4__deadbeef__20260418_143000",
  "run_grade": "exploratory|diagnostic|report-grade",
  "timestamp_utc": "2026-04-18T14:30:00Z",
  "git_sha": "a1b2c3d4e5f6...",
  "git_dirty": false,
  "config_path": "configs/rental_poland_short.toml",
  "config_hash": "deadbeefcafe1234",
  "dataset": {
    "name": "rental_poland_short",
    "source": "remote|local|kaggle|huggingface",
    "path": "data/parquet/rental_poland_short.parquet",
    "n_rows_loaded": 50000,
    "n_rows_after_filter": 48211,
    "target_col": "price"
  },
  "split": {
    "strategy": "random|temporal",
    "test_size": 0.2,
    "test_periods": null,
    "time_col": null,
    "seed": 42,
    "n_train": 38568,
    "n_test": 9643
  },
  "sce": {
    "context_variant": "sce",
    "categorical_mode": "manual|auto",
    "categorical_cols_resolved": ["city", "room_type"],
    "aggregations": ["mean", "median", "std", "count"],
    "min_group_size": 5,
    "use_cross_fitting": true,
    "cross_fit_strategy": "random|time|off",
    "n_folds": 5,
    "include_global_stats": true,
    "include_interactions": true,
    "max_interaction_depth": 2,
    "include_fold_variance": true,
    "fold_variance_features": ["std", "lower", "upper"],
    "include_relative_features": false,
    "n_sce_features": 88,
    "n_context_features": 88
  },
  "model": {
    "type": "xgboost",
    "preset": "default",
    "params": {"n_estimators": 200, "max_depth": 6}
  },
  "metrics": {
    "baseline_rmse": 27368.0,
    "baseline_r2": 0.65,
    "sce_rmse": 22541.0,
    "sce_r2": 0.85,
    "rmse_improvement_pct": 17.64,
    "r2_improvement_pp": 24.49,
    "runtime_seconds": 142.3,
    "n_baseline_features": 12,
    "n_sce_features": 88
  },
  "diagnostics": {
    "permuted_target": null,
    "shuffled_groups": null,
    "crossfit_ab": null,
    "feature_dominance": null
  },
  "promotion": {
    "promoted_to_report_grade": false,
    "blocked_by": ["missing_diagnostic:permuted_target"]
  }
}
```

`diagnostics.*` to `null` jeśli diagnostyka nie była uruchamiana albo `{...result...}` zwrócony przez odpowiedni runner. Promocja do `report-grade` (P1-8) sprawdza `diagnostics` i `feature_dominance`.

---

## 6. Blok P2 — porządek (post-P1)

### P2-9: Propagacja `random_state` z configu do `KFold`

W [`sce/engine.py`](../../sce/engine.py) `KFold(..., random_state=42)` jest hard-coded. Dodać `random_state: int = 42` do `ContextConfig` i użyć tu. Tylko wtedy testy stabilności seedów mają sens.

**DoD:** pole istnieje, używane, 1 test sprawdza że dwa różne seedy → różne wyniki cross-fit.

---

### P2-10: CI test: brak `engine.fit_transform(full_df)` poza scope cross-fit

Dodać `tests/test_no_pre_split_fit.py` — analiza statyczna AST przeszukująca `scripts/` i `sce/` w poszukiwaniu wywołań `engine.fit_transform(...)` poza dozwoloną listą plików (`sce/engine.py`, `tests/`). Każde inne wywołanie → fail testu z linkiem do linii.

**DoD:** test przechodzi na czystym repo. Próbne dodanie `engine.fit_transform(df)` w `scripts/run.py::load_dataset` powoduje fail.

---

### P2-11: Failure case logging

W `run_experiment` i `run_search_experiment` po obliczeniu metryk:
```python
if rmse_improvement_pct < 1.0 or runtime_seconds > 2 * baseline_runtime_seconds:
    append_jsonl(RESULTS_DIR / "failure_cases.jsonl", {
        "run_id": ..., "reason": ..., "metrics": {...}
    })
```

⚠ **DECYZJA WYMAGANA:** `baseline_runtime_seconds` aktualnie nie mierzymy osobno. Trzeba dodać. Sugestia: zmierzyć `time.perf_counter()` wokół `train_configured_model(...)` baseline'a.

✅ **DOPRECYZOWANIE / STATUS (2026-04-18):**
- D8 wdrożone: porównanie runtime jest względem baseline z tego samego runu.
- `run_experiment` i `run_search_experiment` mierzą czas baseline (`baseline_runtime_seconds`) i używają go do reguły `runtime > 2x baseline` w `failure_cases.jsonl`.

**DoD:** plik `results/failure_cases.jsonl` powstaje, format JSONL stabilny, 1 test smoke.

---

## 7. Tracking checklist

Po każdym mergu zmień `[ ]` na `[x]` i dopisz nr PR.

### P0
- [x] **P0-1** Hard guard temporal+random cross-fit (PR #local)
- [x] **P0-2** Train-only categorical encoding (PR #local)
- [x] **P0-3** Train-only feature pruning (PR #local)
- [x] **P0-4** Pełna metadata runa + `--run-grade` flag (PR #local)

### P1
- [x] **P1-5** Permuted target diagnostic (PR #local)
- [x] **P1-6** Shuffled groups diagnostic (PR #local)
- [x] **P1-7** Cross-fit A/B diagnostic (PR #local)
- [x] **P1-8** Feature dominance audit + promocja gating (PR #local)

### P2
- [x] **P2-9** `random_state` propagation (PR #local)
- [x] **P2-10** AST guard `fit_transform` (PR #local)
- [x] **P2-11** Failure case logging (PR #local)

---

## 8. Decyzje WYMAGANE od leada (zbiorczo)

Lista wszystkich `⚠ DECYZJA WYMAGANA` z dokumentu — junior musi to wyjaśnić **przed startem** odpowiedniego ticketu.

| # | Ticket | Decyzja | Sugestia juniora |
|---|---|---|---|
| D1 | P0-1 | ✅ ZREALIZOWANE: `TimeSeriesSplit` (sklearn) vs rolling-window dla temporal cross-fit | rolling-window jako baseline |
| D2 | P0-2 | Sentinel dla unseen categories w test: `-1` vs `NaN` | ✅ ZREALIZOWANE: `NaN` + katalog unseen (`unseen_count`, `unseen_rate`, samples), bez synthetic "unknown" rows |
| D3 | P0-4 | Czy `report-grade` ma być BLOKOWANY przez brak diagnostyk od razu? | ✅ ZREALIZOWANE: hard block + `RuntimeError` + raport blokad w `results/report_grade_blocks.jsonl` |
| D4 | P1-5 | Diagnostyka na subsamplu (~20k) czy pełnym datasecie? | ✅ ZREALIZOWANE: subsample default (`--max-rows 20000`), `--full` opt-in, `report-grade` => full dataset |
| D5 | P1-5 | Próg pass: `sce_advantage_permuted_mean < 1pp` czy `< 0.5pp`? | 1pp |
| D6 | P1-6 | Permutacja wszystkich kolumn naraz vs per-column | ✅ ZREALIZOWANE: oba tryby + breakdown per kolumna w wyniku diagnostyki |
| D7 | P1-8 | Próg dominance `top-3 > 70%` arbitralny — OK? | ✅ ZREALIZOWANE: 70% jako próg startowy + warning, hard block tylko dla `target_global` |
| D8 | P2-11 | Co znaczy „runtime > 2× baseline" — vs poprzedni run czy vs baseline w tym samym runie? | ✅ ZREALIZOWANE: porównanie do baseline z tego samego runu |

---

## 9. Out of scope (nie tykać w tym ticketcie)

- Refaktor `_fit_transform_cross_fitted` na nowy interfejs splittera (osobna inicjatywa, P3+).
- Imputacja braków train-only (`fillna(0)` → train median imputation) — soft warning, nie hard fail. Osobny ticket.
- Pełne pokrycie testów dla `sce/selection.py` (obecnie 17%).
- Dodanie nowych aggregations (counts per fold itd.).
- Refaktor `prepare_features` na sklearn `ColumnTransformer` — może zrobimy później jako P3.

---

## 10. Sekwencja merge'y rekomendowana

1. P0-1 (najmniejszy, łatwo zweryfikować) → unlock dla audytu temporal datasetów.
2. P0-2 + P0-3 (powiązane, mogą być w 1 PR jeśli małe) → poprawia jakość pomiaru baseline'u.
3. P0-4 → metadata pojawia się w nowych runach; **rerun wszystkich datasetów** po tym kroku, bo dopiero teraz mają komplet metadanych.
4. P1-5, P1-6 równolegle → 2 osobne PR-y.
5. P1-7 → szybki, niezależny.
6. P1-8 → na końcu, bo używa danych z search pipeline.
7. P2-9, P2-10, P2-11 — w dowolnej kolejności po P1.

Po P1-8 robimy **pełen sweep** wszystkich configów z `--run-grade=report-grade` i sprawdzamy które datasety przechodzą promocję. Wynik aktualizuje tabelę „Werdykt globalny per dataset" z audytu 2026-04-18.

---

## 11. Co NIE jest w tym dokumencie i kogo o to zapytać

- Decyzje statystyczne / paper-related (np. czy ratio features wracają w jakiejś bezpiecznej formie) → `auditor` agent / autor paper.
- Architektura nowych modułów `scripts/diagnostics/` → tech lead.
- CI integracja (uruchomienie diagnostyk w GitHub Actions) → osobny ticket po P1-8.
