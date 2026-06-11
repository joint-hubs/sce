# Dataset Expansion Candidates

## Goal

Add more datasets that stress SCE on hierarchical structure without overfitting the benchmark suite to one domain.

## Recommendation

Prioritize datasets with:

- a numeric prediction target
- at least 2 to 4 natural hierarchy levels
- enough rows to make split-first evaluation meaningful
- a public or officially downloadable source
- a plausible tabular conversion path for SCE

## Review Of Proposed Kaggle Links

### 1. TS-8: Hierarchical time series

- Source type: competition notebook
- Underlying data: M5 Forecasting - Accuracy
- Verdict: strong candidate
- Why: item, department, category, store, and state create a real multi-level hierarchy with a widely used benchmark
- Caveat: this requires time-aware feature generation and temporal validation, not random splits

### 2. Time series | LGB. Hierarchical daily time-series

- Source type: public notebook over a hierarchical daily series dataset
- Verdict: possible candidate, but lower priority until the raw dataset slug is pinned down and licensing is checked
- Why: it appears aligned with daily hierarchical forecasting, but the notebook metadata is less clear than M5

### 3. Timeseries Forecasting - Hierarchical and Ensemble

- Source type: private dataset notebook
- Verdict: skip
- Why: private input means it is not a stable benchmark for this repo

## Best Immediate Additions

### 1. M5 Forecasting - Accuracy

- Best fit for the links you shared
- Hierarchy: state -> store -> category -> department -> item
- Target: daily unit demand
- Why it is valuable: clean, difficult, standard hierarchical benchmark
- Required repo work: add temporal split support and a preprocessing script that converts the competition files into a supervised table with lag/date features

### 2. Corporacion Favorita Grocery Sales Forecasting

- Stronger than Rossmann for hierarchy depth
- Hierarchy: state/city -> store -> product family -> item
- Target: unit sales
- Why it is valuable: multiple business hierarchies plus calendar effects, promotions, and store metadata
- Required repo work: same time-aware evaluation path as M5

### 3. Walmart Recruiting - Store Sales Forecasting

- Good intermediate benchmark
- Hierarchy: store -> department -> week
- Target: weekly sales
- Why it is valuable: simpler than M5, easier first forecasting benchmark
- Required repo work: temporal split and holiday/promo preprocessing

### 4. Rossmann Store Sales

- Good fallback if we want a simpler public benchmark first
- Hierarchy: state/region-like store metadata -> store -> date
- Target: daily sales
- Why it is valuable: easier pipeline, but weaker hierarchy than M5 or Favorita

## Best Immediate Plan

1. Add provider-aware download support so raw competition or dataset files can be fetched reproducibly.
2. Start with M5 as the first time-series benchmark.
3. Add temporal validation to the runner before claiming results on forecasting data.
4. Add one retail backup benchmark, preferably Favorita or Walmart.

## Cross-Domain Kaggle Shortlist

The next wave should diversify away from real estate and pure retail while still preserving clear grouped structure.

### Tier 1: Best overall fit

#### 1. ASHRAE - Great Energy Predictor III

- Domain: energy / buildings
- Kaggle: https://www.kaggle.com/competitions/ashrae-energy-prediction
- Target: `meter_reading`
- Natural hierarchy: site -> building -> meter type -> timestamp
- Why it fits SCE: strong repeated entities, clear location structure, temporal dynamics, and a numeric target
- Expected preprocessing: join building metadata and weather, then model per row or per building-meter panel
- Strengths: different domain, widely recognized benchmark, good narrative value for rebuttal breadth
- Caveat: large data volume and careful time-aware splits are required

#### 2. Recruit Restaurant Visitor Forecasting

- Domain: hospitality / restaurant operations
- Kaggle: https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting
- Target: daily restaurant visitors
- Natural hierarchy: area -> genre -> restaurant -> date
- Why it fits SCE: explicit grouped entities, reservation logs, restaurant metadata, and a direct forecasting target
- Expected preprocessing: merge visit history, reservations, store metadata, calendar, and weather if used
- Strengths: clearly hierarchical, non-retail, easier to explain than many web or session datasets
- Caveat: still temporal, so it needs the same split discipline as M5

#### 3. Mercari Price Suggestion Challenge

- Domain: marketplace / e-commerce pricing
- Kaggle: https://www.kaggle.com/competitions/mercari-price-suggestion-challenge
- Target: item price
- Natural hierarchy: top category -> subcategory -> leaf category, plus brand -> condition -> seller text context
- Why it fits SCE: tabular regression with an explicit category hierarchy and many grouping candidates
- Expected preprocessing: split `category_name` into hierarchy levels, clean text-derived categoricals, drop very sparse leaves if needed
- Strengths: genuinely different from current datasets, not primarily a time-series problem, and likely the easiest new non-real-estate benchmark for the current library
- Caveat: category hierarchy is user-generated and noisy, so cleanup rules will matter

### Tier 2: Strong but heavier preprocessing

#### 4. Google Analytics Customer Revenue Prediction

- Domain: digital commerce / web analytics
- Kaggle: https://www.kaggle.com/competitions/ga-customer-revenue-prediction
- Target: log revenue per `fullVisitorId`
- Natural hierarchy: country/region/city -> traffic source / channel -> device -> visitor -> session time
- Why it fits SCE: many nested categorical structures and a high-value numeric target
- Expected preprocessing: flatten nested JSON-like columns, aggregate session history carefully, define whether the modeling unit is session or visitor
- Strengths: broadens the paper into web analytics and customer-value prediction
- Caveat: more ETL work than ASHRAE, Recruit, or Mercari

#### 5. Web Traffic Time Series Forecasting

- Domain: internet / content demand
- Kaggle: https://www.kaggle.com/competitions/web-traffic-time-series-forecasting
- Target: daily visits
- Natural hierarchy: project -> access type -> agent -> page -> date
- Why it fits SCE: the competition explicitly mentions hierarchical time-series modeling and has decomposable metadata in page identifiers
- Expected preprocessing: parse page strings into hierarchy columns and construct lag/calendar features
- Strengths: strong domain diversity and a clean hierarchical story
- Caveat: the raw panel is large and page parsing is somewhat bespoke

#### 6. Grupo Bimbo Inventory Demand

- Domain: food distribution / supply chain
- Kaggle: https://www.kaggle.com/competitions/grupo-bimbo-inventory-demand
- Target: demand
- Natural hierarchy: state or depot geography -> route -> channel -> client -> product
- Why it fits SCE: multiple business hierarchies, a direct numeric target, and a classic demand setting
- Expected preprocessing: normalize route/client/product identifiers and potentially engineer time windows if using date fields from the broader release assets
- Strengths: strong hierarchy and a well-known demand benchmark
- Caveat: still adjacent to retail, so it helps less with domain-diversity messaging than ASHRAE or Recruit

## Recommended next picks by goal

### If the goal is strongest domain diversification

1. ASHRAE - Great Energy Predictor III
2. Recruit Restaurant Visitor Forecasting
3. Mercari Price Suggestion Challenge

### If the goal is easiest path to a new benchmark with the current codebase

1. Mercari Price Suggestion Challenge
2. Recruit Restaurant Visitor Forecasting
3. ASHRAE - Great Energy Predictor III

### If the goal is strongest rebuttal story about hierarchical breadth

1. M5 Forecasting - Accuracy
2. ASHRAE - Great Energy Predictor III
3. Recruit Restaurant Visitor Forecasting
4. Mercari Price Suggestion Challenge

## Concrete Plan For The Top 3

The three best next additions are ASHRAE, Recruit, and Mercari.

| Dataset | Domain | SCE Fit | Preprocessing Effort | Why |
|---|---|---|---|---|
| ASHRAE | Energy / buildings | High | Large | Repeated building entities, strong site/building grouping, multiple meter regimes |
| Recruit | Hospitality / restaurants | High | Medium | Clear area/genre/store hierarchy with direct daily demand target |
| Mercari | Marketplace pricing | Medium-High | Medium | Strong category hierarchy and brand/condition groupings without mandatory time-series work |

### 1. ASHRAE - Great Energy Predictor III

- Estimated SCE fit: High
- Why: SCE should work well when there are stable entity groups with shared local behavior. ASHRAE has repeated observations by `building_id`, `site_id`, and `meter`, which is exactly the kind of structure where contextual summaries can help.
- Main hierarchy to use:
  - `site_id -> building_id -> meter`
  - optional side groupings from `primary_use`
- Expected preprocessing effort: Large
  - Estimated effort: about 3 to 5 implementation days
  - Main work: merge `train.csv`, `building_metadata.csv`, and `weather_train.csv`; sort out timestamp alignment; build lag/calendar/weather features; decide the final row grain
- Recommended repo deliverables:
  - `scripts/prepare_ashrae_dataset.py`
  - `configs/ashrae_building_meter_hourly.toml` or a daily aggregated variant
  - tests for joins, lag generation, and temporal split behavior
- Recommended first version:
  - aggregate to daily building-meter rows instead of keeping full hourly resolution
  - keep categorical features compact: `site_id`, `building_id`, `meter`, `primary_use`
  - numeric features: weather summaries, square footage, year built, weekday, month, rolling usage
- Risk level: Medium
  - The biggest risk is ETL complexity rather than poor SCE fit
- Public solution viability: Strong
  - This is a good candidate for a public write-up because the feature pipeline is interpretable and the benchmark is well known

### 2. Recruit Restaurant Visitor Forecasting

- Estimated SCE fit: High
- Why: this is one of the cleanest SCE matches outside retail. Stores live inside area and cuisine groupings, and the target is a local demand signal that should benefit from area- and genre-level context.
- Main hierarchy to use:
  - `air_area_name -> air_genre_name -> air_store_id`
  - time context from `visit_date`
- Expected preprocessing effort: Medium
  - Estimated effort: about 2 to 3 implementation days
  - Main work: merge visits, reservations, store metadata, date features, and optionally weather/holiday tables
- Recommended repo deliverables:
  - `scripts/prepare_recruit_dataset.py`
  - `configs/recruit_restaurant_daily.toml`
  - tests for reservation aggregation and temporal holdout construction
- Recommended first version:
  - model daily visitors per `air_store_id`
  - aggregate reservations into same-day and lead-time features
  - use categorical columns like `air_area_name`, `air_genre_name`, `air_store_id`, weekday, holiday flags
  - start with a 39-day or similar final-period holdout following the competition framing
- Risk level: Low-Medium
  - The joins are manageable and the hierarchy is explicit
- Public solution viability: Strong
  - This is probably the easiest polished end-to-end notebook after Mercari

### 3. Mercari Price Suggestion Challenge

- Estimated SCE fit: Medium-High
- Why: Mercari is not a natural temporal hierarchy, but it has a very usable categorical tree from `category_name` plus `brand_name` and `item_condition_id`. That makes it a good tabular regression test of whether SCE helps in noisy marketplace pricing.
- Main hierarchy to use:
  - `category_lvl_1 -> category_lvl_2 -> category_lvl_3`
  - side groupings from `brand_name` and `item_condition_id`
- Expected preprocessing effort: Medium
  - Estimated effort: about 1.5 to 3 implementation days
  - Main work: split category strings, normalize sparse brands/categories, decide how much text to keep out of scope for the first benchmark
- Recommended repo deliverables:
  - `scripts/prepare_mercari_dataset.py`
  - `configs/mercari_price.toml`
  - tests for category parsing and sparse-category cleanup
- Recommended first version:
  - ignore heavy NLP and build a clean tabular baseline first
  - numeric features: item description length, name length, shipping flag
  - categorical features: category levels, brand, condition
  - use random split first, then optionally add grouped or seller-aware stress tests later
- Risk level: Low
  - Easiest ETL among the three, but category noise may cap gains
- Public solution viability: Very strong
  - Best choice if the goal is a fast, shareable Kaggle-style notebook using SCE as an interpretable feature layer

## Recommended Order Of Implementation

1. Mercari
   - Fastest path to a new cross-domain benchmark
   - Lowest preprocessing burden
   - Good candidate for a public notebook
2. Recruit
   - Best clean hierarchical benchmark after Mercari
   - Strong SCE fit with manageable temporal preprocessing
3. ASHRAE
   - Strongest breadth signal, but most engineering work

## Practical Estimate

- Quick win path: Mercari first, Recruit second
- Strong paper/rebuttal path: Recruit first, ASHRAE second
- Best overall program: Mercari for speed, Recruit for clean hierarchy, ASHRAE for breadth

## Posting / solution-sharing notes

- Posting code or notebooks is realistic for these Kaggle competitions, but raw competition data should still be fetched from Kaggle rather than redistributed from this repo.
- Mercari is especially friendly for a polished public notebook story because the competition explicitly centered kernel reruns.
- ASHRAE is also attractive for a public solution because the competition materials emphasize solution packaging and open-source-style deliverables.
- For repo integration, the safe pattern remains: keep only configs, download scripts, checksums, and deterministic preprocessing code under version control.

## Integration Notes

- Kaggle-backed manifest sources should use one of these forms:
  - `kaggle://competitions/<competition>/<file>`
  - `kaggle://datasets/<owner>/<dataset>/<file>`
- Raw competition data should land in a preprocessing step before becoming the anonymized parquet files used by experiments.
- Do not mix raw competition downloads with the final benchmark parquet files without a deterministic transformation script.