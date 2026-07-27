"""Equity data subpackage: delisting-aware universe registry + loader.

S1.1: ``registry`` (universe configs) + ``loader`` (alive-ticker resolution).
S1.2: ``schema`` (pandera validation) + ``fetch`` (yfinance OHLCV wrapper) +
``EquityDataLoader.fetch_prices`` (partitioned parquet store).
S1.3: ``articles`` layer + ``EquityDataLoader.fetch_articles`` / \
``join_articles_to_prices`` (point-in-time text join via XNYS schedule) +
``equity.diagnostics.published_at_guard`` leakage guard CLI.

The schema/fetch modules are intentionally NOT re-exported here to keep
``import equity`` light (pandera/yfinance are loaded lazily only when
``fetch_prices`` / ``fetch_articles`` are called). Import them explicitly:

    from equity.data.schema import prices_schema, validate_prices
    from equity.data.fetch import fetch_yfinance_ohlcv, fetch_articles_from_seed
"""
