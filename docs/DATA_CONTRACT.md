# Data Contract

Raw CSV (actual columns from `all_albums_full.csv`)
- Artist
- Album
- Year
- Release Date
- Genres
- Critic Score
- User Score
- Avg Track Score
- User Ratings
- Critic Reviews
- Tracks
- Runtime (min)
- Avg Track Runtime (min)
- Label
- Descriptors
- Album URL
- All Artists
- Album Type

Required columns (baseline)
- Artist, Album, Year, Release Date, Genres
- User Score, User Ratings
- Tracks, Runtime (min), Avg Track Runtime (min)
- Album Type, All Artists

Optional columns (imputable or optional)
- Critic Score, Critic Reviews, Avg Track Score
- Descriptors (used for descriptor features; may be sparse)
- Label, Album URL (not used in baseline models)

Canonical names (internal)
- Artist -> Artist
- Album -> Album
- Year -> Year
- Release Date -> Release_Date
- Genres -> Genres
- Critic Score -> Critic_Score
- User Score -> User_Score
- Avg Track Score -> Avg_Track_Score
- User Ratings -> User_Ratings
- Critic Reviews -> Critic_Reviews
- Tracks -> Num_Tracks
- Runtime (min) -> Runtime_Min
- Avg Track Runtime (min) -> Avg_Runtime
- Label -> Label
- Descriptors -> Descriptors
- Album URL -> Album_URL
- All Artists -> All_Artists
- Album Type -> Album_Type

Mapping implementation
- See `src/aoty_pred/data/cleaning.py` for `RAW_TO_CANONICAL`.

Source reference
- See docs/lineage/DATA_LINEAGE_DETAILED.md for full lineage and derived columns.
- See docs/RAW_SCHEMA_SNAPSHOT.md for a snapshot of the raw CSV headers and sample dtypes.

Cleaning rules (baseline)
- Min ratings threshold: default 30.
- Drop rows with missing User Score.
- Drop rows with missing critical numeric fields after repair attempts.
- Record exclusion reasons per row.

Outputs (minimum)
- data/processed/regression_ready.csv
- data/splits/split_manifest.json
- data/processed/imputation_log.csv

Target variable
- User Score (continuous), prediction for next album per artist.
