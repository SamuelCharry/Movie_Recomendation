"""
user_user.py — Genera model_cache.pkl para el modelo User-User.

Uso:
    python user_user.py build

Archivos necesarios en dat/ (mismos que usa Item-Item.py):
    dat/rating.csv   (dataset original MovieLens 20M)
    dat/movie.csv    (catálogo de películas MovieLens)

Genera:
    model_cache.pkl
"""

import sys
import pickle
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans

# =============================================================================
# CONFIGURACION — mismo preprocesamiento que Item-Item.py
# =============================================================================
DAT_DIR     = Path(__file__).parent / "dat"
OUTPUT_FILE = Path(__file__).parent / "model_cache.pkl"

RANDOM_SEED   = 505
SAMPLE_SIZE   = 0.30   # igual que Item-Item.py

MIN_PER_USER  = 20
MIN_PER_MOVIE = 10
N_USERS       = 2000   # reduccion necesaria para RAM en user-user

BEST_K     = 50
BEST_MINK  = 5
BEST_GAMMA = 200


# =============================================================================
# Utilidad: timestamps (igual que Item-Item.py)
# =============================================================================
def _ts_a_fecha(ts_val) -> str:
    if ts_val is None or (isinstance(ts_val, float) and np.isnan(ts_val)):
        return "1970-01-01"
    try:
        unix = int(float(ts_val))
        return datetime.datetime.fromtimestamp(unix, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(ts_val)[:10]


# =============================================================================
# PASO 1 — Cargar datos
# =============================================================================
def cargar_datos():
    print("Cargando datos...")
    rating = pd.read_csv(DAT_DIR / "rating.csv")
    movie  = pd.read_csv(DAT_DIR / "movie.csv")
    print(f"  rating: {len(rating):,} | movie: {len(movie):,}")
    return rating, movie


# =============================================================================
# PASO 2 — Muestreo estratificado (identico a Item-Item.py)
# =============================================================================
def muestreo_estratificado(rating):
    print(f"Muestreo estratificado {int(SAMPLE_SIZE*100)}%...")
    rpu  = rating.groupby("userId").size()
    bins = list(range(0, int(rpu.max()) + 101, 100)) + [float("inf")]
    strata = pd.cut(rpu, bins=bins, include_lowest=True).reset_index()
    strata.columns = ["userId", "stratum"]
    selected = []
    for i, (_, g) in enumerate(strata.groupby("stratum", observed=True)):
        ids = g["userId"].tolist()
        n   = max(1, int(round(SAMPLE_SIZE * len(ids))))
        rng = np.random.default_rng(RANDOM_SEED + i)
        selected.extend([ids[j] for j in rng.choice(len(ids), size=min(n, len(ids)), replace=False)])
    sample = rating[rating["userId"].isin(set(selected))].copy()
    print(f"  {len(selected):,} usuarios | {len(sample):,} ratings")
    return sample


# =============================================================================
# PASO 3 — Filtro adicional para Surprise
# =============================================================================
def filtrar_para_surprise(sample):
    print(f"Filtrando top {N_USERS} usuarios mas activos...")
    uc = sample["userId"].value_counts()
    mc = sample["movieId"].value_counts()
    df = sample[
        sample["userId"].isin(uc[uc >= MIN_PER_USER].index) &
        sample["movieId"].isin(mc[mc >= MIN_PER_MOVIE].index)
    ]
    top = df["userId"].value_counts().head(N_USERS).index
    df  = df[df["userId"].isin(top)].copy()
    density = len(df) / (df.userId.nunique() * df.movieId.nunique())
    print(f"  {len(df):,} ratings | {df.userId.nunique()} usuarios | {df.movieId.nunique()} peliculas | densidad {density:.1%}")
    return df


# =============================================================================
# PASO 4 — Catalogo completo de peliculas (~27k, igual que Item-Item.py)
# =============================================================================
def construir_movies_dict(movie):
    print("Construyendo catalogo de peliculas (~27k)...")
    d = {}
    for _, row in movie.iterrows():
        genres = (
            row["genres"].split("|")
            if pd.notna(row.get("genres", "")) and row["genres"] != "(no genres listed)"
            else []
        )
        d[int(row["movieId"])] = {"title": row["title"], "genres": genres}
    print(f"  {len(d):,} peliculas cargadas")
    return d


# =============================================================================
# PASO 5 — Historial de ratings con ratedAt (igual que Item-Item.py)
# =============================================================================
def construir_user_ratings(rating_sample):
    """
    Incluye ratedAt convertido desde timestamp — igual que Item-Item.py.
    Esto evita inconsistencias al mezclar ambos modelos en main.py.
    """
    print("Construyendo historial de ratings por usuario...")
    user_ratings = {}
    for row in rating_sample.itertuples(index=False):
        uid = int(row.userId)
        if uid not in user_ratings:
            user_ratings[uid] = []
        user_ratings[uid].append({
            "movieId": int(row.movieId),
            "rating":  float(row.rating),
            "ratedAt": _ts_a_fecha(row.timestamp),
        })
    print(f"  {len(user_ratings):,} usuarios en historial")
    return user_ratings


# =============================================================================
# PASO 6 — Entrenar modelo
# =============================================================================
def entrenar(ratings):
    print(f"\nEntrenando pearson_baseline k={BEST_K} gamma={BEST_GAMMA}...")
    reader   = Reader(rating_scale=(0.5, 5.0))
    dataset  = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset = dataset.build_full_trainset()
    algo = KNNWithMeans(
        k=BEST_K, min_k=BEST_MINK,
        sim_options={"name": "pearson_baseline", "user_based": True, "shrinkage": BEST_GAMMA},
        verbose=True
    )
    algo.fit(trainset)
    print("Modelo entrenado.")
    return algo, trainset


# =============================================================================
# PASO 7 — Precalcular vecinos
# =============================================================================
def precalcular_vecinos(algo, trainset):
    print(f"\nPrecalculando vecinos para {trainset.n_users} usuarios...")
    neighbors = {}
    for inner in trainset.all_users():
        raw     = trainset.to_raw_uid(inner)
        sim_row = algo.sim[inner]
        top_idx = np.argsort(sim_row)[::-1][1:BEST_K+1]
        nbs = [
            {"userId": trainset.to_raw_uid(nb), "similarity": round(float(sim_row[nb]), 4)}
            for nb in top_idx if sim_row[nb] > 0
        ]
        neighbors[int(raw)] = nbs
    print(f"  Vecinos listos para {len(neighbors):,} usuarios")
    return neighbors


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "build":
        print("Uso: python user_user.py build")
        sys.exit(1)

    print("=" * 60)
    print("USER-USER — Generando model_cache.pkl")
    print("=" * 60)

    if not (DAT_DIR / "rating.csv").exists():
        print("ERROR: No se encontro dat/rating.csv")
        print("Descarga MovieLens 20M y pon rating.csv y movie.csv en dat/")
        sys.exit(1)

    rating, movie  = cargar_datos()
    sample         = muestreo_estratificado(rating)
    del rating
    ratings        = filtrar_para_surprise(sample)
    # Construir historial ANTES de filtrar — para incluir todos los ratings del sample
    user_ratings   = construir_user_ratings(sample)
    del sample
    movies_dict    = construir_movies_dict(movie)
    algo, trainset = entrenar(ratings)
    neighbors      = precalcular_vecinos(algo, trainset)

    print(f"\nGuardando {OUTPUT_FILE}...")
    pickle.dump({
        "trainset":       trainset,
        "movies_dict":    movies_dict,      # catalogo completo ~27k peliculas
        "user_ratings":   user_ratings,     # historial con ratedAt (consistente con Item-Item)
        "user_neighbors": neighbors,
        "best_model":     algo,
        "best_params": {
            "similarity": "pearson_baseline",
            "k":          BEST_K,
            "min_k":      BEST_MINK,
            "gamma":      BEST_GAMMA,
            "mae":        0.5880,
            "rmse":       0.7718,
        },
    }, open(OUTPUT_FILE, "wb"))

    print(f"\n{'='*60}")
    print(f"model_cache.pkl listo")
    print(f"  Usuarios en modelo   : {trainset.n_users:,}")
    print(f"  Usuarios en historial: {len(user_ratings):,}")
    print(f"  Peliculas en catalogo: {len(movies_dict):,}")
    print(f"  Modelo: pearson_baseline k={BEST_K} gamma={BEST_GAMMA}")
    print(f"{'='*60}")
    print(f"\nAhora corre: python -m uvicorn main:app --reload --port 8000")