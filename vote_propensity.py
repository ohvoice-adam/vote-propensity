#!/usr/bin/env python3
"""
Ohio Voter Propensity Model
============================
Predicts likelihood of voting in the May 5, 2026 primary election.

Features:
  - Ingests any Ohio statewide voter file (CSV)
  - Appends race & gender probabilities via pyethnicity
  - Trains on 2014/2018/2022 federal midterm primaries (panel model)
  - Accounts for precinct activity and voter eligibility
  - GPU-accelerated XGBoost where available
  - Progress indicators throughout
"""

import argparse
import multiprocessing as mp
import os
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

# Suppress onnxruntime TensorRT/CUDA noise before any onnxruntime import
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Constants ───────────────────────────────────────────────────────────────

ELECTION_DATE = date(2026, 5, 5)
MIN_VOTING_AGE = 18

# Federal midterm primaries used as training targets
MIDTERM_PRIMARIES = {
    2014: ("PRIMARY-05/06/2014", date(2014, 5, 6)),
    2018: ("PRIMARY-05/08/2018", date(2018, 5, 8)),
    2022: ("PRIMARY-05/03/2022", date(2022, 5, 3)),
}

# Precincts with < this fraction of eligible voters casting ballots
# are treated as having no contested race (voter not penalized for not voting)
PRECINCT_INACTIVE_THRESHOLD = 0.03

PARTY_MAP = {"R": 0, "D": 1, "L": 2, "G": 3, "N": 4, "": 5}


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Ohio Voter Propensity Model — May 5, 2026 Primary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("voter_file", help="Path to Ohio voter file (CSV)")
    p.add_argument("--output", default="predictions.csv",
                   help="Output CSV path (default: predictions.csv)")
    p.add_argument("--use-party", action="store_true", default=False,
                   help="Include party affiliation as a model feature (off by default for 501c3 compliance)")
    p.add_argument("--no-gpu", dest="gpu", action="store_false", default=True,
                   help="Disable GPU acceleration")
    p.add_argument("--ethnicity-chunksize", type=int, default=10_000,
                   help="Chunk size per worker for pyethnicity calls (default: 10000)")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                   help="Parallel workers for ethnicity prediction (default: cpu_count-1)")
    p.add_argument("--skip-ethnicity", action="store_true",
                   help="Skip race/gender prediction (if already appended)")
    p.add_argument("--no-cache", dest="cache", action="store_false", default=True,
                   help="Disable ethnicity result caching")
    p.add_argument("--n-trees", type=int, default=500,
                   help="Number of XGBoost trees (default: 500)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Directory for checkpoint files to resume failed runs "
                        "(default: <output_stem>.checkpoints/ next to output)")
    return p.parse_args()


# Columns we actually use — everything else is dropped on load to save memory.
_KEEP_DEMO_COLS = {
    "SOS_VOTERID", "COUNTY_NUMBER", "COUNTY_ID",
    "LAST_NAME", "FIRST_NAME",
    "DATE_OF_BIRTH", "REGISTRATION_DATE",
    "VOTER_STATUS", "PARTY_AFFILIATION",
    "RESIDENTIAL_ADDRESS1", "RESIDENTIAL_CITY", "RESIDENTIAL_STATE", "RESIDENTIAL_ZIP",
    "PRECINCT_NAME", "PRECINCT_CODE",
    "CONGRESSIONAL_DISTRICT", "STATE_REPRESENTATIVE_DISTRICT", "STATE_SENATE_DISTRICT",
}

# Low-cardinality string columns encoded as Categorical to save memory
_CATEGORICAL_COLS = {
    "VOTER_STATUS", "PARTY_AFFILIATION", "COUNTY_NUMBER",
    "CONGRESSIONAL_DISTRICT", "STATE_REPRESENTATIVE_DISTRICT", "STATE_SENATE_DISTRICT",
    "PRECINCT_CODE",
}


# ─── Step 1: Load voter file ──────────────────────────────────────────────────

def load_voter_file(path: str) -> pd.DataFrame:
    print("\n[1/6] Loading voter file...")
    path = Path(path)
    if not path.exists():
        sys.exit(f"ERROR: File not found: {path}")

    file_size_mb = path.stat().st_size / 1_048_576
    print(f"      File: {path.name}  ({file_size_mb:,.1f} MB)")

    # Read header to identify election vs. demographic columns
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()

    quoting = 1 if header_line.startswith('"') else 0
    raw_cols = [c.strip().strip('"') for c in header_line.rstrip("\n").split(",")]

    election_col_set = {
        c for c in raw_cols
        if c.startswith("PRIMARY-") or c.startswith("GENERAL-") or c.startswith("SPECIAL-")
    }
    keep_cols = (_KEEP_DEMO_COLS | election_col_set) & set(raw_cols)

    print(f"      Columns: {len(raw_cols)} total → keeping {len(keep_cols)} "
          f"({len(election_col_set)} election + {len(keep_cols) - len(election_col_set)} demo)")

    chunks = []
    chunk_iter = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        usecols=list(keep_cols),
        chunksize=100_000,
        encoding="utf-8",
        encoding_errors="replace",
        quoting=quoting,
    )

    total_rows = 0
    with tqdm(desc="      Reading", unit=" rows", unit_scale=True, ncols=80) as pbar:
        for chunk in chunk_iter:
            chunk.columns = [c.strip().strip('"') for c in chunk.columns]

            # ── Convert election columns to int8 immediately ──
            # Values: 'R','D','X','' etc. → 1 if non-empty, 0 if empty.
            ecols_in_chunk = [c for c in chunk.columns if c in election_col_set]
            for col in ecols_in_chunk:
                chunk[col] = chunk[col].str.strip().ne("").astype(np.int8)

            # ── Encode low-cardinality strings as Categorical ──
            for col in _CATEGORICAL_COLS:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype("category")

            chunks.append(chunk)
            total_rows += len(chunk)
            pbar.update(len(chunk))

    df = pd.concat(chunks, ignore_index=True)

    mem_mb = df.memory_usage(deep=True).sum() / 1_048_576
    print(f"      Loaded {total_rows:,} voters  |  {len(df.columns)} columns  |  "
          f"{mem_mb:,.0f} MB in memory")
    return df


# ─── Step 2: Parse election columns ──────────────────────────────────────────

def identify_election_columns(df: pd.DataFrame):
    """Return (election_cols, col_dates) where col_dates maps col→date."""
    election_cols = []
    col_dates = {}
    for col in df.columns:
        for prefix in ("PRIMARY-", "GENERAL-", "SPECIAL-"):
            if col.startswith(prefix):
                date_str = col[len(prefix):]  # MM/DD/YYYY
                try:
                    dt = datetime.strptime(date_str, "%m/%d/%Y").date()
                    election_cols.append(col)
                    col_dates[col] = dt
                except ValueError:
                    pass
                break
    # Sort chronologically
    election_cols.sort(key=lambda c: col_dates[c])
    return election_cols, col_dates


def voted(series: pd.Series) -> pd.Series:
    """Return boolean: voter cast ballot. Handles int8 (post-load) and string columns."""
    if series.dtype == np.int8 or series.dtype == bool:
        return series.astype(bool)
    return series.str.strip().ne("")


def is_primary_col(col: str) -> bool:
    return col.startswith("PRIMARY-")


def is_general_col(col: str) -> bool:
    return col.startswith("GENERAL-")


# ─── Step 3: Core demographic feature extraction ──────────────────────────────

def extract_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Parse DOB, registration date, party, location."""
    print("\n[2/6] Extracting demographics...")

    df["dob"] = pd.to_datetime(df["DATE_OF_BIRTH"], errors="coerce")
    df["reg_date"] = pd.to_datetime(df["REGISTRATION_DATE"], errors="coerce")
    # Free the raw string columns — datetimes are half the size
    df.drop(columns=["DATE_OF_BIRTH", "REGISTRATION_DATE"], errors="ignore", inplace=True)

    # Age at target election (2026-05-05); cap at 120 for bad DOB data
    target = pd.Timestamp(ELECTION_DATE)
    df["age_2026"] = ((target - df["dob"]).dt.days / 365.25).clip(lower=0, upper=120)

    # Birth year (for gender prediction)
    df["birth_year"] = df["dob"].dt.year.fillna(1970).astype(int)

    # Party encoding
    df["party_code"] = df["PARTY_AFFILIATION"].str.strip().map(
        lambda x: PARTY_MAP.get(x, 5)
    )

    # ZIP (ZCTA proxy)
    df["zip5"] = df["RESIDENTIAL_ZIP"].str.strip().str[:5]

    # County as integer
    df["county_num"] = pd.to_numeric(df["COUNTY_NUMBER"], errors="coerce").fillna(0).astype(int)

    # Congressional / state legislative districts
    df["cong_dist"] = pd.to_numeric(df["CONGRESSIONAL_DISTRICT"], errors="coerce").fillna(0).astype(int)
    df["state_rep_dist"] = pd.to_numeric(df["STATE_REPRESENTATIVE_DISTRICT"], errors="coerce").fillna(0).astype(int)
    df["state_sen_dist"] = pd.to_numeric(df["STATE_SENATE_DISTRICT"], errors="coerce").fillna(0).astype(int)

    # Precinct key
    df["precinct_key"] = df["county_num"].astype(str) + "_" + df["PRECINCT_CODE"].str.strip()

    print(f"      {df['dob'].notna().sum():,} valid birth dates parsed")
    print(f"      Age range: {df['age_2026'].min():.0f}–{df['age_2026'].max():.0f}")
    return df


# ─── Step 4: Race & gender prediction via pyethnicity ────────────────────────

def _suppress_c_stderr():
    """Context manager that silences C-level stderr (e.g. onnxruntime TensorRT noise)."""
    import contextlib, os

    @contextlib.contextmanager
    def _cm():
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull_fd, 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            os.close(devnull_fd)

    return _cm()


def _ethnicity_worker(args: tuple) -> tuple:
    """
    Top-level worker function (must be module-level for pickling).
    Receives a chunk of voter data, returns (chunk_idx, race_array, gender_array).
    race_array shape: (n, 4)  columns: asian, black, hispanic, white
    gender_array shape: (n,)  pct_female
    """
    import os
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")
    import warnings
    warnings.filterwarnings("ignore")

    # Patch onnxruntime before pyethnicity loads it: force CPU-only provider,
    # preventing the TensorRT/CUDA probe that emits the noisy error messages.
    import onnxruntime as _ort
    _orig_init = _ort.InferenceSession.__init__
    def _cpu_only_init(self, path_or_bytes, sess_options=None, providers=None, **kw):
        _orig_init(self, path_or_bytes, sess_options,
                   providers=["CPUExecutionProvider"], **kw)
    _ort.InferenceSession.__init__ = _cpu_only_init

    import pyethnicity
    import numpy as np

    chunk_idx, first, last, zips, birth_years = args
    n = len(first)
    race_arr   = np.full((n, 4), 0.25)
    gender_arr = np.full(n, 0.5)

    # ── Race ──
    try:
        race_df = pyethnicity.predict_race(first_name=first, last_name=last, zcta=zips)
        if hasattr(race_df, "to_pandas"):
            race_df = race_df.to_pandas()
        col_map = {}
        for col in race_df.columns:
            cl = col.lower()
            if "asian" in cl:   col_map[col] = "asian"
            elif "black" in cl: col_map[col] = "black"
            elif "hisp"  in cl: col_map[col] = "hispanic"
            elif "white" in cl: col_map[col] = "white"
        race_df = race_df.rename(columns=col_map)
        for i, col in enumerate(["asian", "black", "hispanic", "white"]):
            if col in race_df.columns:
                race_arr[:, i] = race_df[col].fillna(0.25).values
    except Exception:
        pass  # keep uniform priors on failure

    # ── Gender ──
    try:
        sex_df = pyethnicity.predict_sex_ssa(
            first_name=first, min_year=birth_years, max_year=birth_years
        )
        if hasattr(sex_df, "collect"):
            sex_df = sex_df.collect()
        if hasattr(sex_df, "to_pandas"):
            sex_df = sex_df.to_pandas()
        female_col = next((c for c in sex_df.columns if "female" in c.lower()), None)
        if female_col:
            gender_arr = sex_df[female_col].fillna(0.5).values
    except Exception:
        pass  # keep 0.5 on failure

    return chunk_idx, race_arr, gender_arr


def _cache_path(voter_file: str) -> Path:
    p = Path(voter_file)
    return p.with_name(p.stem + "_ethnicity_cache.parquet")


def predict_ethnicity(
    df: pd.DataFrame,
    voter_file: str,
    chunksize: int = 10_000,
    n_workers: int = 1,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Append race and gender probability columns, with caching and multiprocessing."""
    race_cols = ["asian", "black", "hispanic", "white", "pct_female"]

    # ── Try loading from cache ──────────────────────────────────────────────
    cache_file = _cache_path(voter_file)
    if use_cache and cache_file.exists():
        print(f"\n[3/6] Loading ethnicity from cache: {cache_file.name}")
        try:
            cache = pd.read_parquet(cache_file)
            # Merge on voter ID; predict only missing rows
            df = df.merge(
                cache[["SOS_VOTERID"] + race_cols],
                on="SOS_VOTERID",
                how="left",
            )
            n_missing = df["asian"].isna().sum()
            if n_missing == 0:
                print(f"      Cache hit: {len(df):,} voters (0 to predict)")
                df["pct_female"] = df["pct_female"].fillna(0.5)
                return df
            print(f"      Partial cache: {len(df) - n_missing:,} cached, {n_missing:,} to predict")
            # Only predict the missing subset
            missing_mask = df["asian"].isna()
            sub = df[missing_mask].copy()
        except Exception as e:
            print(f"      Cache read failed ({e}); predicting from scratch")
            for col in race_cols:
                df[col] = np.nan
            missing_mask = pd.Series(True, index=df.index)
            sub = df.copy()
    else:
        for col in race_cols:
            df[col] = np.nan
        missing_mask = pd.Series(True, index=df.index)
        sub = df.copy()
        cache = None

    n = len(sub)
    n_workers = min(n_workers, max(1, n // chunksize + 1))
    print(f"\n[3/6] Predicting race & gender ({n:,} voters, "
          f"{n_workers} worker{'s' if n_workers > 1 else ''}, "
          f"chunk size {chunksize:,})...")

    # Build work items: list of (chunk_idx, first, last, zips, birth_years)
    work = []
    indices = list(range(0, n, chunksize)) + [n]
    for i in range(len(indices) - 1):
        lo, hi = indices[i], indices[i + 1]
        chunk = sub.iloc[lo:hi]
        work.append((
            i,
            chunk["FIRST_NAME"].str.strip().tolist(),
            chunk["LAST_NAME"].str.strip().tolist(),
            chunk["zip5"].tolist(),
            chunk["birth_year"].tolist(),
        ))

    # ── Run predictions (parallel or serial) ───────────────────────────────
    race_arr   = np.full((n, 4), 0.25)
    gender_arr = np.full(n, 0.5)

    if n_workers > 1:
        ctx = mp.get_context("fork")  # fork is safe on Linux; avoids re-loading model
        with ctx.Pool(n_workers) as pool:
            with tqdm(total=n, desc="      Ethnicity", unit=" voters", ncols=80) as pbar:
                for chunk_idx, r_arr, g_arr in pool.imap_unordered(
                    _ethnicity_worker, work
                ):
                    lo = chunk_idx * chunksize
                    hi = lo + len(g_arr)
                    race_arr[lo:hi]   = r_arr
                    gender_arr[lo:hi] = g_arr
                    pbar.update(hi - lo)
    else:
        with tqdm(total=n, desc="      Ethnicity", unit=" voters", ncols=80) as pbar:
            for item in work:
                chunk_idx, r_arr, g_arr = _ethnicity_worker(item)
                lo = chunk_idx * chunksize
                hi = lo + len(g_arr)
                race_arr[lo:hi]   = r_arr
                gender_arr[lo:hi] = g_arr
                pbar.update(hi - lo)

    # Write results back into sub
    for i, col in enumerate(["asian", "black", "hispanic", "white"]):
        sub[col] = race_arr[:, i]
    sub["pct_female"] = gender_arr

    # Merge sub results back into full df
    df.update(sub[["asian", "black", "hispanic", "white", "pct_female"]])
    for col in race_cols:
        if col == "pct_female":
            df[col] = df[col].fillna(0.5)
        else:
            df[col] = df[col].fillna(0.25)

    # ── Save / update cache ─────────────────────────────────────────────────
    if use_cache:
        try:
            new_cache = df[["SOS_VOTERID"] + race_cols].copy()
            if cache is not None and len(cache):
                # Merge old cache rows that aren't in current df
                old_only = cache[~cache["SOS_VOTERID"].isin(new_cache["SOS_VOTERID"])]
                new_cache = pd.concat([new_cache, old_only], ignore_index=True)
            new_cache.to_parquet(cache_file, index=False)
            print(f"      Cache saved: {cache_file.name}  ({len(new_cache):,} records)")
        except Exception as e:
            print(f"      WARNING: could not save cache ({e})")

    print(f"      Race/gender appended. Sample means — "
          f"white:{df['white'].mean():.2f}  black:{df['black'].mean():.2f}  "
          f"hispanic:{df['hispanic'].mean():.2f}  asian:{df['asian'].mean():.2f}")
    return df


# ─── Step 5: Voting history features & precinct activity ─────────────────────

def was_eligible(dob_series: pd.Series, reg_series: pd.Series, election_dt: date) -> pd.Series:
    """True if voter was 18+ and registered before the election."""
    edt = pd.Timestamp(election_dt)
    age_ok  = ((edt - dob_series).dt.days / 365.25) >= MIN_VOTING_AGE
    reg_ok  = reg_series <= edt
    return age_ok & reg_ok


def detect_precinct_activity(
    df: pd.DataFrame,
    election_col: str,
    election_dt: date,
    col_dates: dict,
) -> pd.Series:
    """
    Returns a boolean Series: True if the precinct had a contested election.
    Heuristic: ≥3% of eligible voters in the precinct cast a ballot.
    """
    elig = was_eligible(df["dob"], df["reg_date"], election_dt)
    voted_flag = voted(df[election_col]) & elig

    precinct_elig  = elig.groupby(df["precinct_key"]).sum()
    precinct_voted = voted_flag.groupby(df["precinct_key"]).sum()

    precinct_rate = (precinct_voted / precinct_elig.clip(lower=1)).rename("rate")
    active = (precinct_rate >= PRECINCT_INACTIVE_THRESHOLD) & (precinct_voted >= 3)

    return df["precinct_key"].map(active).fillna(False)


def build_voting_history_features(
    df: pd.DataFrame,
    as_of_date: date,
    election_cols: list,
    col_dates: dict,
) -> pd.DataFrame:
    """
    Build voter-level voting history features as of a specific date.
    Returns a DataFrame aligned with df's index.
    """
    prior_cols = [c for c in election_cols if col_dates[c] < as_of_date]
    prior_primary = [c for c in prior_cols if is_primary_col(c)]
    prior_general = [c for c in prior_cols if is_general_col(c)]

    edt = pd.Timestamp(as_of_date)

    # Eligibility at this date
    age_at = ((edt - df["dob"]).dt.days / 365.25).clip(lower=0)
    reg_ok  = df["reg_date"] <= edt
    eligible_now = (age_at >= MIN_VOTING_AGE) & reg_ok

    # For each prior election, was voter eligible?
    def _eligible_for(col):
        cdt = pd.Timestamp(col_dates[col])
        age_ok = ((cdt - df["dob"]).dt.days / 365.25) >= MIN_VOTING_AGE
        reg_ok2 = df["reg_date"] <= cdt
        return age_ok & reg_ok2

    # Compute eligible prior primaries and votes
    n_elig_primary = pd.Series(0, index=df.index)
    n_voted_primary = pd.Series(0, index=df.index)
    last_primary_vote_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    for col in prior_primary:
        elig = _eligible_for(col)
        v = voted(df[col]) & elig
        n_elig_primary  += elig.astype(int)
        n_voted_primary += v.astype(int)
        cdt = pd.Timestamp(col_dates[col])
        last_primary_vote_date = last_primary_vote_date.where(~v, cdt)

    n_elig_general = pd.Series(0, index=df.index)
    n_voted_general = pd.Series(0, index=df.index)
    last_general_vote_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    for col in prior_general:
        elig = _eligible_for(col)
        v = voted(df[col]) & elig
        n_elig_general  += elig.astype(int)
        n_voted_general += v.astype(int)
        cdt = pd.Timestamp(col_dates[col])
        last_general_vote_date = last_general_vote_date.where(~v, cdt)

    # Rates (eligible-adjusted)
    primary_rate = n_voted_primary / n_elig_primary.clip(lower=1)
    general_rate = n_voted_general / n_elig_general.clip(lower=1)

    # Years since last vote (large value = never voted)
    years_since_primary = ((edt - last_primary_vote_date).dt.days / 365.25).fillna(99)
    years_since_general = ((edt - last_general_vote_date).dt.days / 365.25).fillna(99)

    # Voted in most recent prior primary / general?
    voted_last_primary = pd.Series(False, index=df.index)
    voted_last_general = pd.Series(False, index=df.index)
    if prior_primary:
        last_p = prior_primary[-1]
        voted_last_primary = voted(df[last_p]) & _eligible_for(last_p)
    if prior_general:
        last_g = prior_general[-1]
        voted_last_general = voted(df[last_g]) & _eligible_for(last_g)

    # Voted in the previous midterm primary (4 years prior)
    voted_prev_midterm = pd.Series(False, index=df.index)
    midterm_primaries_prior = [
        c for c in prior_primary
        if abs((col_dates[c].year - as_of_date.year) - 4) < 2
        and col_dates[c].month in (3, 4, 5)
    ]
    if midterm_primaries_prior:
        mp = midterm_primaries_prior[-1]
        voted_prev_midterm = voted(df[mp]) & _eligible_for(mp)

    age_at_election = age_at

    return pd.DataFrame(
        {
            "age":                   age_at_election,
            "n_elig_primary":        n_elig_primary,
            "n_voted_primary":       n_voted_primary,
            "primary_rate":          primary_rate.where(n_elig_primary > 0, np.nan),
            "n_elig_general":        n_elig_general,
            "n_voted_general":       n_voted_general,
            "general_rate":          general_rate.where(n_elig_general > 0, np.nan),
            "years_since_primary":   years_since_primary,
            "years_since_general":   years_since_general,
            "voted_last_primary":    voted_last_primary.astype(int),
            "voted_last_general":    voted_last_general.astype(int),
            "voted_prev_midterm":    voted_prev_midterm.astype(int),
            "eligible_now":          eligible_now.astype(int),
        },
        index=df.index,
    )


def build_panel_dataset(
    df: pd.DataFrame,
    election_cols: list,
    col_dates: dict,
    use_party: bool = False,
) -> tuple:
    """
    Build training panel from 2014, 2018, 2022 midterm primaries.
    Returns (X, y, weights) arrays.
    """
    print("\n[4/6] Building training panel (2014 / 2018 / 2022 midterm primaries)...")

    X_parts, y_parts, w_parts = [], [], []

    for year, (col, edate) in tqdm(
        MIDTERM_PRIMARIES.items(), desc="      Elections", ncols=80
    ):
        if col not in df.columns:
            tqdm.write(f"      WARNING: Column '{col}' not found — skipping {year}")
            continue

        # Detect eligible voters
        elig = was_eligible(df["dob"], df["reg_date"], edate)

        # Detect precinct activity
        active_precinct = detect_precinct_activity(df, col, edate, col_dates)

        # Target: voted in this primary (only count if precinct was active)
        y = (voted(df[col]) & elig).astype(int)

        # Weight: 0 if precinct was inactive (exclude from loss)
        w = (elig & active_precinct).astype(float)

        # Only include eligible voters for training
        mask = elig & active_precinct
        if mask.sum() == 0:
            tqdm.write(f"      WARNING: No eligible+active voters for {year}")
            continue

        tqdm.write(
            f"      {year}: {mask.sum():,} eligible voters in active precincts  |  "
            f"turnout: {y[mask].mean():.1%}"
        )

        # Voting history features
        hist = build_voting_history_features(df, edate, election_cols, col_dates)

        # Static demographic features
        feat = pd.DataFrame(
            {
                "year":           year,
                "county_num":     df["county_num"],
                "cong_dist":      df["cong_dist"],
                "state_rep_dist": df["state_rep_dist"],
                "state_sen_dist": df["state_sen_dist"],
                **( {"party_code": df["party_code"]} if use_party else {} ),
                "asian":          df["asian"],
                "black":          df["black"],
                "hispanic":       df["hispanic"],
                "white":          df["white"],
                "pct_female":     df["pct_female"],
            },
            index=df.index,
        )
        feat = pd.concat([feat, hist], axis=1)

        X_parts.append(feat[mask])
        y_parts.append(y[mask])
        w_parts.append(w[mask])

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    w = pd.concat(w_parts, ignore_index=True)

    # Fill NaN primary_rate / general_rate with 0 for new voters
    X["primary_rate"] = X["primary_rate"].fillna(0)
    X["general_rate"] = X["general_rate"].fillna(0)

    print(f"      Panel rows: {len(X):,}  |  Overall turnout: {y.mean():.1%}")
    return X, y, w


# ─── Step 6: Model training ───────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series, w: pd.Series, use_gpu: bool, n_trees: int):
    """Train XGBoost binary classifier."""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    print(f"\n[5/6] Training XGBoost model ({n_trees} trees)...")

    # Detect GPU
    device = "cpu"
    if use_gpu:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                device = "cuda"
                print("      GPU detected — using CUDA acceleration")
            else:
                print("      No CUDA GPU found — using CPU")
        except Exception:
            print("      GPU check failed — using CPU")
    else:
        print("      GPU disabled — using CPU")

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X, y, w, test_size=0.1, random_state=42, stratify=y
    )

    scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    params = dict(
        n_estimators=n_trees,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        device=device,
        early_stopping_rounds=30,
    )

    # Progress callback (XGBoost ≥ 2.0 supports callbacks in constructor)
    pbar = tqdm(total=n_trees, desc="      Training", unit=" trees", ncols=80)

    class _ProgressCB(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            pbar.update(1)
            return False

    try:
        model = xgb.XGBClassifier(**params, callbacks=[_ProgressCB()])
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False,
        )
    except Exception as exc:
        if "cuda" in str(exc).lower() or "gpu" in str(exc).lower():
            pbar.close()
            print(f"      CUDA unavailable ({exc}); falling back to CPU")
            device = "cpu"
            params["device"] = "cpu"
            pbar = tqdm(total=n_trees, desc="      Training (CPU)", unit=" trees", ncols=80)
            model = xgb.XGBClassifier(**params, callbacks=[_ProgressCB()])
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[w_val],
                verbose=False,
            )
        else:
            pbar.close()
            raise
    finally:
        pbar.close()

    # Validation AUC
    from sklearn.metrics import roc_auc_score
    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    best_iter = model.best_iteration if hasattr(model, "best_iteration") else n_trees
    print(f"      Best iteration: {best_iter}  |  Validation AUC: {val_auc:.4f}")

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("      Top 10 features by importance:")
    for feat, imp in fi.head(10).items():
        print(f"        {feat:30s}  {imp:.4f}")

    return model


# ─── Step 7: Predict for 2026 ─────────────────────────────────────────────────

def predict_2026(
    df: pd.DataFrame,
    model,
    election_cols: list,
    col_dates: dict,
    use_party: bool = False,
) -> pd.DataFrame:
    """Generate 2026 primary propensity scores for all active voters."""
    print("\n[6/6] Generating 2026 primary propensity scores...")

    # Eligibility for 2026
    elig_2026 = was_eligible(df["dob"], df["reg_date"], ELECTION_DATE)
    active_status = df["VOTER_STATUS"].str.strip().str.upper().isin(["ACTIVE", "A"])

    # Build features as of 2026 election date
    hist = build_voting_history_features(df, ELECTION_DATE, election_cols, col_dates)

    # Precinct historical activity: what fraction of past primaries was each precinct active?
    all_primary_cols = [c for c in election_cols if is_primary_col(c)]
    precinct_active_counts = pd.Series(0.0, index=df.index)
    precinct_total_counts  = pd.Series(0.0, index=df.index)

    for col in tqdm(all_primary_cols, desc="      Precinct history", ncols=80, leave=False):
        cdt = col_dates[col]
        active = detect_precinct_activity(df, col, cdt, col_dates)
        elig   = was_eligible(df["dob"], df["reg_date"], cdt)
        precinct_active_counts += (active & elig).astype(float)
        precinct_total_counts  += elig.astype(float)

    precinct_activity_rate = (
        precinct_active_counts / precinct_total_counts.clip(lower=1)
    )

    feat = pd.DataFrame(
        {
            "year":                  2026,
            "county_num":            df["county_num"],
            "cong_dist":             df["cong_dist"],
            "state_rep_dist":        df["state_rep_dist"],
            "state_sen_dist":        df["state_sen_dist"],
            **( {"party_code": df["party_code"]} if use_party else {} ),
            "asian":                 df["asian"],
            "black":                 df["black"],
            "hispanic":              df["hispanic"],
            "white":                 df["white"],
            "pct_female":            df["pct_female"],
        },
        index=df.index,
    )
    feat = pd.concat([feat, hist], axis=1)
    feat["primary_rate"] = feat["primary_rate"].fillna(0)
    feat["general_rate"] = feat["general_rate"].fillna(0)

    # Score in chunks for progress display
    n = len(feat)
    chunk = 100_000
    scores = np.empty(n)
    with tqdm(total=n, desc="      Scoring", unit=" voters", ncols=80) as pbar:
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            scores[lo:hi] = model.predict_proba(feat.iloc[lo:hi])[:, 1]
            pbar.update(hi - lo)

    results = pd.DataFrame(
        {
            "SOS_VOTERID":           df["SOS_VOTERID"],
            "LAST_NAME":             df["LAST_NAME"],
            "FIRST_NAME":            df["FIRST_NAME"],
            "DATE_OF_BIRTH":         df["dob"].dt.strftime("%m/%d/%Y"),
            "PARTY_AFFILIATION":     df["PARTY_AFFILIATION"],
            "RESIDENTIAL_ZIP":       df["RESIDENTIAL_ZIP"],
            "PRECINCT_CODE":         df["PRECINCT_CODE"],
            "COUNTY_NUMBER":         df["COUNTY_NUMBER"],
            "VOTER_STATUS":          df["VOTER_STATUS"],
            "asian":                 df["asian"].round(4),
            "black":                 df["black"].round(4),
            "hispanic":              df["hispanic"].round(4),
            "white":                 df["white"].round(4),
            "pct_female":            df["pct_female"].round(4),
            "age_2026":              hist["age"].round(1),
            "primary_rate":          feat["primary_rate"].round(4),
            "general_rate":          feat["general_rate"].round(4),
            "n_elig_primary":        hist["n_elig_primary"].astype(int),
            "n_voted_primary":       hist["n_voted_primary"].astype(int),
            "precinct_activity_rate": precinct_activity_rate.round(4),
            "eligible_2026":         elig_2026.astype(int),
            "active_status":         active_status.astype(int),
            "propensity_score":      scores.round(4),
        }
    )

    # Adjust scores downward for voters in historically inactive precincts
    # (they may not have a race to vote in — inform user via flag)
    results["low_precinct_activity"] = (precinct_activity_rate < 0.5).astype(int)

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = datetime.now()

    import gc

    out_path = Path(args.output)
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = out_path.parent / (out_path.stem + ".checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    DF_CKPT    = ckpt_dir / "df_after_ethnicity.parquet"
    MODEL_CKPT = ckpt_dir / "model.ubj"

    print("=" * 70)
    print("  Ohio Voter Propensity Model — May 5, 2026 Primary")
    print(f"  Run started: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print("=" * 70)

    # ── Steps 1–4: Load + demographics + ethnicity ───────────────────────────
    if DF_CKPT.exists():
        print(f"\n[1-4/6] Resuming from checkpoint: {DF_CKPT.name}")
        df = pd.read_parquet(DF_CKPT)
        election_cols, col_dates = identify_election_columns(df)
        print(f"      Loaded {len(df):,} voters, "
              f"{len(election_cols)} election columns "
              f"({col_dates[election_cols[0]]} – {col_dates[election_cols[-1]]})")
    else:
        # 1. Load
        df = load_voter_file(args.voter_file)

        # 2. Demographics
        df = extract_demographics(df)

        # 3. Election columns
        election_cols, col_dates = identify_election_columns(df)
        print(f"\n      Found {len(election_cols)} election columns "
              f"({col_dates[election_cols[0]]} – {col_dates[election_cols[-1]]})")

        # 4. Race/gender (unless skipped)
        if args.skip_ethnicity:
            print("\n[3/6] Skipping ethnicity prediction (--skip-ethnicity)")
            for col in ["asian", "black", "hispanic", "white", "pct_female"]:
                if col not in df.columns:
                    df[col] = 0.25 if col != "pct_female" else 0.5
        else:
            df = predict_ethnicity(
                df,
                voter_file=args.voter_file,
                chunksize=args.ethnicity_chunksize,
                n_workers=args.workers,
                use_cache=args.cache,
            )

        print(f"\n      Saving checkpoint → {DF_CKPT.name}")
        df.to_parquet(DF_CKPT, index=True)

    # ── Step 5: Build panel & train ──────────────────────────────────────────
    if MODEL_CKPT.exists():
        import xgboost as xgb
        print(f"\n[5/6] Resuming from checkpoint: {MODEL_CKPT.name}")
        model = xgb.XGBClassifier()
        model.load_model(MODEL_CKPT)
    else:
        if not args.use_party:
            print("      Party affiliation excluded (use --use-party to include)")
        X, y, w = build_panel_dataset(df, election_cols, col_dates, use_party=args.use_party)

        model = train_model(X, y, w, use_gpu=args.gpu, n_trees=args.n_trees)

        print(f"\n      Saving checkpoint → {MODEL_CKPT.name}")
        model.save_model(MODEL_CKPT)

        # Free panel data — XGBoost has its own internal copy
        del X, y, w
        gc.collect()

    # ── Step 6: Predict (needs election cols still in df) ────────────────────
    results = predict_2026(df, model, election_cols, col_dates, use_party=args.use_party)

    # Free election columns from df — no longer needed after prediction
    df.drop(columns=election_cols, errors="ignore", inplace=True)
    gc.collect()

    # Save
    results.to_csv(out_path, index=False)
    elapsed = (datetime.now() - t0).total_seconds()

    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.0f}s  |  Output: {out_path}  ({len(results):,} voters)")
    print(f"  Checkpoints in: {ckpt_dir}  (safe to delete after a successful run)")

    elig = (results["eligible_2026"] == 1) & (results["active_status"] == 1)
    scored = results[elig.values]
    if len(scored):
        print(f"\n  Eligible active voters: {len(scored):,}")
        print(f"  Mean propensity score:  {scored['propensity_score'].mean():.3f}")
        print(f"  Score distribution:")
        for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
            n = ((scored["propensity_score"] >= lo) & (scored["propensity_score"] < hi)).sum()
            print(f"    {lo:.1f}–{hi:.1f}:  {n:>8,}  ({n/len(scored):.1%})")
    print("=" * 70)


if __name__ == "__main__":
    main()
