"""
Item-Item.py — Genera item_item_cache.pkl para el modelo Item-Item.

Uso:
    python Item-Item.py build

Archivos necesarios en dat/:
    dat/rating.csv   (dataset original MovieLens 20M)
    dat/movie.csv    (catálogo de películas MovieLens)

Genera:
    item_item_cache.pkl
"""

import os
import sys
import json
import pickle
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =============================================================================
# SECCIÓN 1 — Constantes y rutas
# =============================================================================
DAT_DIR    = Path(__file__).parent / "dat"
CACHE_PATH = Path(__file__).parent / "item_item_cache.pkl"

RANDOM_SEED          = 505
SAMPLE_SIZE          = 0.3
MIN_RATINGS_PELICULA = 20

# Variables globales del modelo
media_por_item: Optional[pd.Series]   = None
rating_promedio_global: Optional[float] = None


# =============================================================================
# Utilidad: conversión de timestamps
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
# SECCIÓN 2 — Funciones del pipeline
# =============================================================================
def cargar_datos():
    print("Cargando datos...")
    rating = pd.read_csv(DAT_DIR / "rating.csv")
    movie  = pd.read_csv(DAT_DIR / "movie.csv")
    print(f"  rating: {len(rating):,} filas | movie: {len(movie):,} filas")
    return rating, movie


def muestreo_estratificado(rating: pd.DataFrame, movie: pd.DataFrame):
    print("Realizando muestreo estratificado...")
    ratings_per_user = rating.groupby("userId").size()
    max_ratings  = int(ratings_per_user.max())
    bins_strata  = list(range(0, max_ratings + 101, 100)) + [float("inf")]
    user_stratum = pd.cut(ratings_per_user, bins=bins_strata, include_lowest=True)
    users_with_stratum = user_stratum.reset_index()
    users_with_stratum.columns = ["userId", "stratum"]

    selected_user_ids = []
    for stratum_idx, (_, group) in enumerate(users_with_stratum.groupby("stratum", observed=True)):
        user_ids = group["userId"].tolist()
        n        = len(user_ids)
        n_sample = max(1, int(round(SAMPLE_SIZE * n))) if n > 0 else 0
        if n_sample > 0:
            rng = np.random.default_rng(RANDOM_SEED + stratum_idx)
            idx = rng.choice(n, size=min(n_sample, n), replace=False)
            selected_user_ids.extend([user_ids[i] for i in idx])

    selected_set  = set(selected_user_ids)
    rating_sample = rating[rating["userId"].isin(selected_set)].copy()
    movie_ids_en_muestra = rating_sample["movieId"].unique()
    movie_sample  = movie[movie["movieId"].isin(movie_ids_en_muestra)].copy()
    print(f"  Usuarios: {len(selected_user_ids):,} de {rating['userId'].nunique():,}")
    print(f"  Ratings : {len(rating_sample):,}")
    return rating_sample, movie_sample


def dividir_train_test(rating_sample: pd.DataFrame):
    print("Dividiendo train/test (70/30)...")
    train_set, test_set = train_test_split(
        rating_sample, test_size=0.3, random_state=RANDOM_SEED
    )
    print(f"  Train: {len(train_set):,} | Test: {len(test_set):,}")
    return train_set, test_set


def filtrar_y_construir_matriz(train_set: pd.DataFrame):
    print(f"Filtrando peliculas con >= {MIN_RATINGS_PELICULA} ratings...")
    peliculas_frecuentes = (
        train_set.groupby("movieId").size()[lambda s: s >= MIN_RATINGS_PELICULA].index
    )
    filtrado = train_set[train_set["movieId"].isin(peliculas_frecuentes)].copy()
    print(f"  Peliculas frecuentes: {len(peliculas_frecuentes):,}")
    print("Construyendo matriz usuario-item...")
    matriz = filtrado.pivot_table(index="userId", columns="movieId", values="rating")
    n_u, n_i = matriz.shape
    densidad = matriz.notna().sum().sum() / (n_u * n_i) * 100
    print(f"  {n_u:,} usuarios x {n_i:,} peliculas | Densidad: {densidad:.2f}%")
    return matriz, peliculas_frecuentes


def calcular_similitud_coseno(matriz_ui: pd.DataFrame) -> pd.DataFrame:
    print("Calculando similitud coseno...")
    m = matriz_ui.fillna(0).values.T.astype(np.float32)
    return pd.DataFrame(cosine_similarity(m), index=matriz_ui.columns, columns=matriz_ui.columns)


def calcular_similitud_pearson(matriz_ui: pd.DataFrame) -> pd.DataFrame:
    print("Calculando similitud Pearson...")
    media = matriz_ui.mean(axis=0)
    centrada = matriz_ui.subtract(media, axis=1).fillna(0).values.T.astype(np.float32)
    return pd.DataFrame(cosine_similarity(centrada), index=matriz_ui.columns, columns=matriz_ui.columns)


def calcular_similitud_jaccard(matriz_ui: pd.DataFrame):
    print("Calculando similitud Jaccard...")
    binaria = matriz_ui.notna().values.astype(np.float32)
    inter   = binaria.T @ binaria
    conteo  = np.diag(inter)
    union   = conteo[:, None] + conteo[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    co_ratings = inter.astype(np.int32)
    return (pd.DataFrame(sim,        index=matriz_ui.columns, columns=matriz_ui.columns),
            pd.DataFrame(co_ratings, index=matriz_ui.columns, columns=matriz_ui.columns))


def calcular_pesos_mclaughlin(matriz_co_ratings: pd.DataFrame, gamma: float) -> pd.DataFrame:
    n = matriz_co_ratings.values.astype(float)
    return pd.DataFrame(np.minimum(n, gamma) / gamma,
                        index=matriz_co_ratings.index, columns=matriz_co_ratings.columns)


# =============================================================================
# SECCIÓN 3 — Funciones del modelo (para evaluación)
# =============================================================================
def predecir_rating(id_usuario, id_item, matriz_ui, sim, modo="top_k", k=20, umbral=0.1):
    if id_item not in sim.index or id_usuario not in matriz_ui.index:
        return rating_promedio_global
    rated = matriz_ui.loc[id_usuario].dropna().drop(id_item, errors="ignore")
    if rated.empty:
        return rating_promedio_global
    comunes = rated.index.intersection(sim.columns)
    if comunes.empty:
        return rating_promedio_global
    sims = sim.loc[id_item, comunes]
    vecinos = sims[sims > 0].nlargest(k) if modo == "top_k" else sims[sims >= umbral]
    if vecinos.empty:
        return rating_promedio_global
    media_obj = media_por_item.get(id_item, rating_promedio_global)
    num = sum(s * (rated[nb] - media_por_item.get(nb, rating_promedio_global)) for nb, s in vecinos.items())
    den = vecinos.abs().sum()
    if den == 0:
        return rating_promedio_global
    return float(np.clip(media_obj + num / den, 0.5, 5.0))


def evaluar_modelo(test, matriz_ui, sim, k=20, muestra=2000):
    validos = test[test["userId"].isin(matriz_ui.index) & test["movieId"].isin(sim.index)]
    if len(validos) > muestra:
        validos = validos.sample(muestra, random_state=RANDOM_SEED)
    reales, preds = [], []
    for row in validos.itertuples(index=False):
        preds.append(predecir_rating(row.userId, row.movieId, matriz_ui, sim, k=k))
        reales.append(row.rating)
    reales, preds = np.array(reales), np.array(preds)
    return {
        "rmse": float(np.sqrt(mean_squared_error(reales, preds))),
        "mae":  float(mean_absolute_error(reales, preds)),
        "n":    len(reales),
    }


# =============================================================================
# SECCIÓN 4 — Build del cache
# =============================================================================
def construir_cache():
    global media_por_item, rating_promedio_global

    rating, movie = cargar_datos()
    rating_sample, movie_sample = muestreo_estratificado(rating, movie)
    del rating

    train_set, test_set = dividir_train_test(rating_sample)

    media_por_item         = train_set.groupby("movieId")["rating"].mean()
    rating_promedio_global = float(train_set["rating"].mean())
    print(f"  Rating promedio global: {rating_promedio_global:.4f}")

    matriz, _ = filtrar_y_construir_matriz(train_set)

    sim_coseno  = calcular_similitud_coseno(matriz)
    sim_pearson = calcular_similitud_pearson(matriz)
    sim_jaccard, co_ratings = calcular_similitud_jaccard(matriz)

    print("Calculando pesos McLaughlin (gamma=25)...")
    pesos_25 = calcular_pesos_mclaughlin(co_ratings, gamma=25)

    # Catalogo completo de peliculas (~27k)
    print("Construyendo movies_dict (~27k peliculas)...")
    movies_dict = {}
    for _, fila in movie.iterrows():
        genres = (fila["genres"].split("|")
                  if pd.notna(fila["genres"]) and fila["genres"] != "(no genres listed)"
                  else [])
        movies_dict[int(fila["movieId"])] = {"title": fila["title"], "genres": genres}

    # Ratings por usuario
    print("Construyendo user_ratings...")
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

    # Fechas de incorporacion
    user_join_dates = {}
    for uid, ts in rating_sample.groupby("userId")["timestamp"].min().items():
        user_join_dates[int(uid)] = _ts_a_fecha(ts)

    print(f"\nGuardando cache en {CACHE_PATH}...")
    cache = {
        "matriz_usuario_item":    matriz,
        "sim_coseno":             sim_coseno,
        "sim_pearson":            sim_pearson,
        "sim_jaccard":            sim_jaccard,
        "matriz_co_ratings":      co_ratings,
        "media_por_item":         media_por_item,
        "rating_promedio_global": rating_promedio_global,
        "pesos_25":               pesos_25,
        "movies_dict":            movies_dict,
        "user_ratings":           user_ratings,
        "user_join_dates":        user_join_dates,
        "train_set":              train_set,
        "test_set":               test_set,
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print("Cache guardado.")
    return cache


# =============================================================================
# SECCIÓN 5 — Main
# =============================================================================
if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "build"

    if modo == "build":
        print("=" * 64)
        print("ITEM-ITEM — Generando item_item_cache.pkl")
        print("=" * 64)

        if not (DAT_DIR / "rating.csv").exists():
            print("ERROR: No se encontro dat/rating.csv")
            sys.exit(1)

        cache = construir_cache()

        n_u, n_i = cache["matriz_usuario_item"].shape
        densidad = cache["matriz_usuario_item"].notna().sum().sum() / (n_u * n_i) * 100

        print("\n" + "=" * 64)
        print("RESUMEN")
        print(f"  Usuarios : {n_u:,}")
        print(f"  Peliculas: {n_i:,}")
        print(f"  Densidad : {densidad:.2f}%")
        print()
        for nombre, key in [("Coseno","sim_coseno"),("Pearson","sim_pearson"),("Jaccard","sim_jaccard")]:
            res = evaluar_modelo(cache["test_set"], cache["matriz_usuario_item"], cache[key])
            print(f"  {nombre}: RMSE={res['rmse']:.4f} | MAE={res['mae']:.4f} (n={res['n']})")
        print("=" * 64)
        print(f"\nAhora corre: python -m uvicorn main:app --reload --port 8000")
    else:
        print("Uso: python Item-Item.py build")
        sys.exit(1)