# Parquet files stored here
# 
# Structure:
# - rental_poland_short.parquet (committed to repo)
# - rental_poland_long.parquet (committed to repo)
# 
# Remote datasets (downloaded on demand):
# - rental_uae_contracts.parquet
# - sales_uae_transactions.parquet

# Anonymization Notes
# - rental_poland_short.parquet retains: city, room_type, property_type, capacity,
#   is_superhost, host_rating, host_review_count, rating_cleanliness, review_count,
#   price_display, price_PLN_per_night.
# - rental_poland_long.parquet retains: city, district, region, subdistrict, area_sqm,
#   price, price_currency, price_per_sqm, rooms, estate_type, transaction_type,
#   floor_number, development_estate_type, development_floor.
# - Dropped columns include direct identifiers and free-text fields such as ids, URLs,
#   seller names, street addresses, images, and descriptions.
