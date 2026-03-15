# =============================================================================
# Item-Item.py — Sistema de recomendación filtrado colaborativo ítem-ítem
# Pipeline completo + servidor FastAPI
#
# Uso:
#   python Item-Item.py build      → construye y guarda item_item_cache.pkl
#   uvicorn Item-Item:app --reload --port 8000  → inicia el servidor
# =============================================================================

# =============================================================================
# SECCIÓN 0 — Imports
# =============================================================================
import os
import sys
import json
import pickle
import asyncio
import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =============================================================================
# SECCIÓN 1 — Constantes y rutas
# =============================================================================
DAT_DIR       = Path(__file__).parent / 'dat'
CACHE_PATH    = Path(__file__).parent / 'item_item_cache.pkl'
EXTRA_PATH    = Path(__file__).parent / 'user_ratings_extra.json'
USUARIOS_PATH = Path(__file__).parent / 'nuevos_usuarios.json'

RANDOM_SEED          = 505
SAMPLE_SIZE          = 0.3
MIN_RATINGS_PELICULA = 20

# Variables globales del modelo — se inicializan al cargar el caché en el lifespan
media_por_item: Optional[pd.Series]  = None
rating_promedio_global: Optional[float] = None
cache_global: Optional[dict]         = None
lock_io: Optional[asyncio.Lock]      = None


# =============================================================================
# Utilidad: conversión de timestamps
# =============================================================================

def _ts_a_fecha(ts_val) -> str:
    """
    Convierte un timestamp a string 'YYYY-MM-DD'.
    Acepta:
      - Entero Unix (ej. 972172489)
      - String datetime (ej. '2000-11-21 15:34:49')
      - String date (ej. '2000-11-21')
    """
    if ts_val is None or (isinstance(ts_val, float) and np.isnan(ts_val)):
        return '1970-01-01'
    try:
        unix = int(float(ts_val))
        return datetime.datetime.fromtimestamp(unix, tz=datetime.timezone.utc).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        # Es un string datetime/date
        return str(ts_val)[:10]


# =============================================================================
# SECCIÓN 2 — Funciones del pipeline (solo se ejecutan al construir el caché)
# =============================================================================

def cargar_datos():
    """Carga rating.csv y movie.csv desde DAT_DIR."""
    print("Cargando datos...")
    rating = pd.read_csv(DAT_DIR / 'rating.csv')
    movie  = pd.read_csv(DAT_DIR / 'movie.csv')
    print(f"  rating: {len(rating):,} filas | movie: {len(movie):,} filas")
    return rating, movie


def muestreo_estratificado(rating: pd.DataFrame, movie: pd.DataFrame):
    """
    Realiza muestreo estratificado del SAMPLE_SIZE (30%) de usuarios.
    Estratos definidos en bins de 100 ratings por usuario.
    Retorna (rating_sample, movie_sample).
    """
    print("Realizando muestreo estratificado...")
    ratings_per_user = rating.groupby("userId").size()
    max_ratings  = int(ratings_per_user.max())
    bins_strata  = list(range(0, max_ratings + 101, 100)) + [float('inf')]
    user_stratum = pd.cut(ratings_per_user, bins=bins_strata, include_lowest=True)

    users_with_stratum = (
        user_stratum
        .reset_index()
    )
    users_with_stratum.columns = ['userId', 'stratum']

    selected_user_ids = []
    for stratum_idx, (stratum_id, group) in enumerate(
        users_with_stratum.groupby("stratum", observed=True)
    ):
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

    print(f"  Usuarios seleccionados: {len(selected_user_ids):,} de {rating['userId'].nunique():,}")
    print(f"  Ratings en muestra: {len(rating_sample):,}")
    return rating_sample, movie_sample


def dividir_train_test(rating_sample: pd.DataFrame):
    """Divide en train/test con proporción 70/30."""
    print("Dividiendo train/test (70/30)...")
    train_set, test_set = train_test_split(
        rating_sample,
        test_size=0.3,
        random_state=RANDOM_SEED,
    )
    print(f"  Train: {len(train_set):,} | Test: {len(test_set):,}")
    return train_set, test_set


def filtrar_y_construir_matriz(train_set: pd.DataFrame):
    """
    Filtra películas con < MIN_RATINGS_PELICULA y construye la matriz usuario-ítem
    mediante pivot_table. Retorna (matriz_usuario_item, peliculas_frecuentes).
    """
    print(f"Filtrando películas con >= {MIN_RATINGS_PELICULA} ratings...")
    peliculas_frecuentes = (
        train_set.groupby('movieId').size()[lambda s: s >= MIN_RATINGS_PELICULA].index
    )
    entrenamiento_filtrado = train_set[
        train_set['movieId'].isin(peliculas_frecuentes)
    ].copy()

    print(f"  Películas frecuentes: {len(peliculas_frecuentes):,}")
    print("Construyendo matriz usuario-ítem...")
    matriz_usuario_item = entrenamiento_filtrado.pivot_table(
        index='userId', columns='movieId', values='rating'
    )
    n_u, n_i = matriz_usuario_item.shape
    densidad = matriz_usuario_item.notna().sum().sum() / (n_u * n_i) * 100
    print(f"  Forma: {n_u:,} usuarios × {n_i:,} películas | Densidad: {densidad:.2f}%")
    return matriz_usuario_item, peliculas_frecuentes


def calcular_similitud_coseno(matriz_ui: pd.DataFrame) -> pd.DataFrame:
    """Calcula la similitud coseno entre ítems."""
    print("Calculando similitud coseno...")
    matriz_rellena = matriz_ui.fillna(0).values.T.astype(np.float32)
    similitud = cosine_similarity(matriz_rellena)
    return pd.DataFrame(similitud, index=matriz_ui.columns, columns=matriz_ui.columns)


def calcular_similitud_pearson(matriz_ui: pd.DataFrame) -> pd.DataFrame:
    """Calcula la similitud Pearson entre ítems (coseno sobre matriz centrada por ítem)."""
    print("Calculando similitud Pearson...")
    media_items     = matriz_ui.mean(axis=0)
    matriz_centrada = matriz_ui.subtract(media_items, axis=1)
    matriz_centrada = matriz_centrada.fillna(0).values.T.astype(np.float32)
    similitud       = cosine_similarity(matriz_centrada)
    return pd.DataFrame(similitud, index=matriz_ui.columns, columns=matriz_ui.columns)


def calcular_similitud_jaccard(matriz_ui: pd.DataFrame):
    """
    Calcula la similitud Jaccard entre ítems basada en co-ocurrencia de ratings.
    Retorna (sim_jaccard DataFrame, matriz_co_ratings DataFrame).
    """
    print("Calculando similitud Jaccard...")
    binaria      = matriz_ui.notna().values.astype(np.float32)
    interseccion = binaria.T @ binaria
    conteo_items = np.diag(interseccion)
    union        = conteo_items[:, None] + conteo_items[None, :] - interseccion

    with np.errstate(divide='ignore', invalid='ignore'):
        similitud = np.where(union > 0, interseccion / union, 0.0)

    matriz_co_ratings = interseccion.astype(np.int32)
    sim_jaccard = pd.DataFrame(similitud,        index=matriz_ui.columns, columns=matriz_ui.columns)
    df_co_ratings = pd.DataFrame(matriz_co_ratings, index=matriz_ui.columns, columns=matriz_ui.columns)
    return sim_jaccard, df_co_ratings


def calcular_pesos_mclaughlin(matriz_co_ratings: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """
    Calcula pesos de significancia según McLaughlin.
    pesos[i,j] = min(co_ratings[i,j], gamma) / gamma
    """
    n     = matriz_co_ratings.values.astype(float)
    pesos = np.minimum(n, gamma) / gamma
    return pd.DataFrame(pesos, index=matriz_co_ratings.index, columns=matriz_co_ratings.columns)


# =============================================================================
# SECCIÓN 3 — Gestión del caché
# =============================================================================

def construir_cache():
    """
    Ejecuta el pipeline completo y serializa el caché en CACHE_PATH.
    Llama con: python Item-Item.py build
    """
    global media_por_item, rating_promedio_global

    # 1. Carga
    rating, movie = cargar_datos()

    # 2. Muestreo estratificado
    rating_sample, movie_sample = muestreo_estratificado(rating, movie)
    del rating  # liberar RAM

    # 3. Split
    train_set, test_set = dividir_train_test(rating_sample)

    # 4. Variables globales del modelo
    media_por_item         = train_set.groupby('movieId')['rating'].mean()
    rating_promedio_global = float(train_set['rating'].mean())
    print(f"  Rating promedio global: {rating_promedio_global:.4f}")

    # 5. Matriz usuario-ítem
    matriz_usuario_item, _ = filtrar_y_construir_matriz(train_set)

    # 6. Similitudes (una a la vez para ahorrar RAM)
    sim_coseno  = calcular_similitud_coseno(matriz_usuario_item)
    sim_pearson = calcular_similitud_pearson(matriz_usuario_item)
    sim_jaccard, matriz_co_ratings = calcular_similitud_jaccard(matriz_usuario_item)

    # 7. Pesos McLaughlin gamma=25
    print("Calculando pesos McLaughlin (gamma=25)...")
    pesos_25 = calcular_pesos_mclaughlin(matriz_co_ratings, gamma=25)

    # 8. Diccionario de películas — TODOS los ~27k para búsqueda completa
    print("Construyendo movies_dict (todos ~27k películas)...")
    movies_dict = {}
    for _, fila in movie.iterrows():
        genres_list = (
            fila['genres'].split('|')
            if pd.notna(fila['genres']) and fila['genres'] != '(no genres listed)'
            else []
        )
        movies_dict[int(fila['movieId'])] = {
            'title':  fila['title'],
            'genres': genres_list,
        }

    # 9. Diccionario de ratings por usuario (desde rating_sample completo)
    print("Construyendo user_ratings...")
    user_ratings: dict = {}
    for row in rating_sample.itertuples(index=False):
        uid = int(row.userId)
        if uid not in user_ratings:
            user_ratings[uid] = []
        user_ratings[uid].append({
            'movieId': int(row.movieId),
            'rating':  float(row.rating),
            'ratedAt': _ts_a_fecha(row.timestamp),
        })

    # 10. Fechas de incorporación (timestamp mínimo por usuario → YYYY-MM-DD)
    print("Calculando fechas de incorporación...")
    user_join_dates: dict = {}
    min_ts = rating_sample.groupby('userId')['timestamp'].min()
    for uid, ts in min_ts.items():
        user_join_dates[int(uid)] = _ts_a_fecha(ts)

    # 11. Serializar
    print(f"\nGuardando caché en {CACHE_PATH} ...")
    cache = {
        'matriz_usuario_item':   matriz_usuario_item,
        'sim_coseno':            sim_coseno,
        'sim_pearson':           sim_pearson,
        'sim_jaccard':           sim_jaccard,
        'matriz_co_ratings':     matriz_co_ratings,
        'media_por_item':        media_por_item,
        'rating_promedio_global': rating_promedio_global,
        'pesos_25':              pesos_25,
        'movies_dict':           movies_dict,
        'user_ratings':          user_ratings,
        'user_join_dates':       user_join_dates,
        'train_set':             train_set,
        'test_set':              test_set,
    }
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f, protocol=4)
    print("¡Caché guardado!")
    return cache


def cargar_cache() -> dict:
    """Carga el caché desde disco. Lanza FileNotFoundError si no existe."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el caché en {CACHE_PATH}.\n"
            "Ejecuta primero:  python Item-Item.py build"
        )
    print(f"Cargando caché desde {CACHE_PATH} (puede tardar 30-60 s)...")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    print("¡Caché cargado en memoria!")
    return cache


# =============================================================================
# SECCIÓN 4 — Funciones del modelo
# =============================================================================

def predecir_rating(
    id_usuario,
    id_item,
    matriz_ui: pd.DataFrame,
    matriz_similitud: pd.DataFrame,
    modo_vecinos: str = 'top_k',
    vecinos_k: int = 20,
    umbral_similitud: float = 0.1,
    pesos_significancia: Optional[pd.DataFrame] = None,
) -> float:
    """
    Predice el rating del usuario sobre un ítem usando filtrado colaborativo ítem-ítem.
    Usa las variables globales media_por_item y rating_promedio_global.
    """
    if id_item not in matriz_similitud.index or id_usuario not in matriz_ui.index:
        return rating_promedio_global

    ratings_usuario = matriz_ui.loc[id_usuario].dropna().drop(id_item, errors='ignore')
    if ratings_usuario.empty:
        return rating_promedio_global

    items_comunes = ratings_usuario.index.intersection(matriz_similitud.columns)
    if items_comunes.empty:
        return rating_promedio_global

    similitudes = matriz_similitud.loc[id_item, items_comunes].copy()

    if pesos_significancia is not None:
        items_en_pesos = items_comunes.intersection(pesos_significancia.columns)
        if not items_en_pesos.empty:
            pesos = pesos_significancia.loc[id_item, items_en_pesos]
            similitudes.loc[items_en_pesos] = similitudes.loc[items_en_pesos] * pesos

    if modo_vecinos == 'top_k':
        vecinos = similitudes[similitudes > 0].nlargest(vecinos_k)
    elif modo_vecinos == 'umbral':
        vecinos = similitudes[similitudes >= umbral_similitud]
    else:
        raise ValueError("modo_vecinos debe ser 'top_k' o 'umbral'")

    if vecinos.empty:
        return rating_promedio_global

    media_item_objetivo = media_por_item.get(id_item, rating_promedio_global)
    numerador   = 0.0
    denominador = 0.0

    for id_vecino, sim in vecinos.items():
        media_vecino    = media_por_item.get(id_vecino, rating_promedio_global)
        rating_centrado = ratings_usuario[id_vecino] - media_vecino
        numerador   += sim * rating_centrado
        denominador += abs(sim)

    if denominador == 0:
        return rating_promedio_global

    prediccion = media_item_objetivo + (numerador / denominador)
    return float(np.clip(prediccion, 0.5, 5.0))


def predecir_rating_con_ratings(
    id_item,
    ratings_usuario_series: pd.Series,
    sim_matrix: pd.DataFrame,
    modo_vecinos: str = 'top_k',
    vecinos_k: int = 20,
    umbral_similitud: float = 0.1,
    pesos_significancia: Optional[pd.DataFrame] = None,
) -> float:
    """
    Variante de predecir_rating que acepta directamente la Series de ratings del usuario.
    Útil para usuarios nuevos o que tienen ratings extra fuera de la matriz.
    """
    if id_item not in sim_matrix.index:
        return rating_promedio_global

    ratings_usuario = ratings_usuario_series.dropna().drop(id_item, errors='ignore')
    if ratings_usuario.empty:
        return rating_promedio_global

    items_comunes = ratings_usuario.index.intersection(sim_matrix.columns)
    if items_comunes.empty:
        return rating_promedio_global

    similitudes = sim_matrix.loc[id_item, items_comunes].copy()

    if pesos_significancia is not None:
        items_en_pesos = items_comunes.intersection(pesos_significancia.columns)
        if not items_en_pesos.empty:
            pesos = pesos_significancia.loc[id_item, items_en_pesos]
            similitudes.loc[items_en_pesos] = similitudes.loc[items_en_pesos] * pesos

    if modo_vecinos == 'top_k':
        vecinos = similitudes[similitudes > 0].nlargest(vecinos_k)
    elif modo_vecinos == 'umbral':
        vecinos = similitudes[similitudes >= umbral_similitud]
    else:
        raise ValueError("modo_vecinos debe ser 'top_k' o 'umbral'")

    if vecinos.empty:
        return rating_promedio_global

    media_item_objetivo = media_por_item.get(id_item, rating_promedio_global)
    numerador   = 0.0
    denominador = 0.0

    for id_vecino, sim in vecinos.items():
        media_vecino    = media_por_item.get(id_vecino, rating_promedio_global)
        rating_centrado = ratings_usuario[id_vecino] - media_vecino
        numerador   += sim * rating_centrado
        denominador += abs(sim)

    if denominador == 0:
        return rating_promedio_global

    prediccion = media_item_objetivo + (numerador / denominador)
    return float(np.clip(prediccion, 0.5, 5.0))


def predecir_rating_explicado(
    id_item,
    ratings_usuario_series: pd.Series,
    sim_matrix: pd.DataFrame,
    modo_vecinos: str = 'top_k',
    vecinos_k: int = 20,
    umbral_similitud: float = 0.1,
    pesos_significancia: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Igual a predecir_rating_con_ratings pero también retorna el detalle de los vecinos usados.
    Retorna: {
        'prediccion': float,
        'vecinos': [(id_vecino, sim, rating_usuario, media_vecino), ...]
    }
    """
    if id_item not in sim_matrix.index:
        return {'prediccion': rating_promedio_global, 'vecinos': []}

    ratings_usuario = ratings_usuario_series.dropna().drop(id_item, errors='ignore')
    if ratings_usuario.empty:
        return {'prediccion': rating_promedio_global, 'vecinos': []}

    items_comunes = ratings_usuario.index.intersection(sim_matrix.columns)
    if items_comunes.empty:
        return {'prediccion': rating_promedio_global, 'vecinos': []}

    similitudes = sim_matrix.loc[id_item, items_comunes].copy()

    if pesos_significancia is not None:
        items_en_pesos = items_comunes.intersection(pesos_significancia.columns)
        if not items_en_pesos.empty:
            pesos = pesos_significancia.loc[id_item, items_en_pesos]
            similitudes.loc[items_en_pesos] = similitudes.loc[items_en_pesos] * pesos

    if modo_vecinos == 'top_k':
        vecinos_sel = similitudes[similitudes > 0].nlargest(vecinos_k)
    elif modo_vecinos == 'umbral':
        vecinos_sel = similitudes[similitudes >= umbral_similitud]
    else:
        return {'prediccion': rating_promedio_global, 'vecinos': []}

    if vecinos_sel.empty:
        return {'prediccion': rating_promedio_global, 'vecinos': []}

    media_item_objetivo = media_por_item.get(id_item, rating_promedio_global)
    numerador   = 0.0
    denominador = 0.0
    vecinos_detalle = []

    for id_vecino, sim in vecinos_sel.items():
        media_vecino    = media_por_item.get(id_vecino, rating_promedio_global)
        rating_u        = float(ratings_usuario[id_vecino])
        rating_centrado = rating_u - media_vecino
        numerador   += sim * rating_centrado
        denominador += abs(sim)
        vecinos_detalle.append((int(id_vecino), float(sim), rating_u, float(media_vecino)))

    if denominador == 0:
        return {'prediccion': rating_promedio_global, 'vecinos': vecinos_detalle}

    prediccion = media_item_objetivo + (numerador / denominador)
    return {
        'prediccion': float(np.clip(prediccion, 0.5, 5.0)),
        'vecinos':    vecinos_detalle,
    }


def recomendar_por_item(
    id_item,
    matriz_similitud: pd.DataFrame,
    info_peliculas: pd.DataFrame,
    top_n: int = 10,
):
    """Imprime las top_n películas más similares al ítem dado."""
    if id_item not in matriz_similitud.index:
        print(f'El item {id_item} no está en la matriz de similitud.')
        return

    titulo_orig = info_peliculas.loc[
        info_peliculas['movieId'] == id_item, 'title'
    ].values
    titulo_orig = titulo_orig[0] if len(titulo_orig) else str(id_item)

    similares = (
        matriz_similitud[id_item]
        .drop(id_item)
        .sort_values(ascending=False)
        .head(top_n)
    )
    print(f'Recomendaciones ítem-ítem de: {titulo_orig}')
    for rank, (id_similar, sim) in enumerate(similares.items(), start=1):
        titulo = info_peliculas.loc[
            info_peliculas['movieId'] == id_similar, 'title'
        ].values
        titulo = titulo[0] if len(titulo) else str(id_similar)
        print(f'{rank:>2}. {titulo[:48]:<48}  sim={sim:.4f}')


def evaluar_modelo(
    conjunto_prueba: pd.DataFrame,
    matriz_ui: pd.DataFrame,
    matriz_similitud: pd.DataFrame,
    modo_vecinos: str = 'top_k',
    vecinos_k: int = 20,
    umbral_similitud: float = 0.1,
    pesos_significancia=None,
    muestra_max: int = 2000,
    semilla: int = RANDOM_SEED,
) -> dict:
    """Evalúa el modelo sobre el conjunto de prueba. Retorna RMSE, MAE y n_evaluados."""
    usuarios_validos = set(matriz_ui.index)
    items_validos    = set(matriz_similitud.index)

    prueba = conjunto_prueba[
        conjunto_prueba['userId'].isin(usuarios_validos) &
        conjunto_prueba['movieId'].isin(items_validos)
    ]
    if len(prueba) > muestra_max:
        prueba = prueba.sample(muestra_max, random_state=semilla)

    reales    = []
    predichos = []
    for fila in prueba.itertuples(index=False):
        pred = predecir_rating(
            fila.userId, fila.movieId,
            matriz_ui, matriz_similitud,
            modo_vecinos=modo_vecinos,
            vecinos_k=vecinos_k,
            umbral_similitud=umbral_similitud,
            pesos_significancia=pesos_significancia,
        )
        reales.append(fila.rating)
        predichos.append(pred)

    reales    = np.array(reales)
    predichos = np.array(predichos)
    rmse = float(np.sqrt(mean_squared_error(reales, predichos)))
    mae  = float(mean_absolute_error(reales, predichos))

    return {
        'rmse':        rmse,
        'mae':         mae,
        'n_evaluados': len(reales),
        'reales':      reales,
        'predichos':   predichos,
    }


# =============================================================================
# SECCIÓN 5 — Helpers de persistencia
# =============================================================================

def cargar_ratings_extra() -> dict:
    """Carga user_ratings_extra.json. Retorna {userId_int: [...]}."""
    if not EXTRA_PATH.exists():
        return {}
    with open(EXTRA_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


async def guardar_rating_extra(user_id: int, movie_id: int, rating_val: float, rated_at: str):
    """Agrega un rating a user_ratings_extra.json de forma segura (async lock)."""
    async with lock_io:
        data = cargar_ratings_extra()
        if user_id not in data:
            data[user_id] = []
        data[user_id].append({
            'movieId': movie_id,
            'rating':  rating_val,
            'ratedAt': rated_at,
        })
        with open(EXTRA_PATH, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in data.items()}, f, ensure_ascii=False)


def cargar_nuevos_usuarios() -> list:
    """Carga nuevos_usuarios.json. Retorna lista de dicts."""
    if not USUARIOS_PATH.exists():
        return []
    with open(USUARIOS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


async def guardar_nuevo_usuario(data: dict):
    """Guarda un nuevo usuario en nuevos_usuarios.json de forma segura."""
    async with lock_io:
        usuarios = cargar_nuevos_usuarios()
        usuarios.append(data)
        with open(USUARIOS_PATH, 'w', encoding='utf-8') as f:
            json.dump(usuarios, f, ensure_ascii=False)


# =============================================================================
# SECCIÓN 6 — App FastAPI + CORS + lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache_global, media_por_item, rating_promedio_global, lock_io
    lock_io                = asyncio.Lock()
    cache_global           = cargar_cache()
    media_por_item         = cache_global['media_por_item']
    rating_promedio_global = cache_global['rating_promedio_global']
    yield


app = FastAPI(
    title="Movie Recommendation API — Ítem-Ítem",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers internos del servidor ---

def _verificar_cache():
    if cache_global is None:
        raise HTTPException(
            status_code=503,
            detail="Caché no cargado. Ejecuta primero: python Item-Item.py build",
        )


def _get_sim_matrix(similarity: str) -> pd.DataFrame:
    """Selecciona la matriz de similitud según el parámetro del frontend."""
    mapa = {
        'cosine':  cache_global['sim_coseno'],
        'pearson': cache_global['sim_pearson'],
        'jaccard': cache_global['sim_jaccard'],
    }
    return mapa.get(similarity, cache_global['sim_pearson'])


def _construir_ratings_row(user_id: int) -> pd.Series:
    """
    Construye la Series de ratings de un usuario combinando la matriz del caché
    con cualquier rating extra guardado en disco.
    """
    matriz = cache_global['matriz_usuario_item']
    if user_id in matriz.index:
        row = matriz.loc[user_id].copy()
    else:
        row = pd.Series(dtype=float)

    for r in cargar_ratings_extra().get(user_id, []):
        row[r['movieId']] = r['rating']

    return row


def _fallback_populares(limit: int, all_rated_ids: set, movies_dict: dict) -> list:
    """Retorna las películas más populares (por rating promedio) no vistas por el usuario."""
    populares   = cache_global['media_por_item'].nlargest(limit * 5).index
    recomendaciones = []
    rank = 1
    for mid in populares:
        if int(mid) in all_rated_ids:
            continue
        info = movies_dict.get(int(mid), {'title': f'Movie {mid}', 'genres': []})
        recomendaciones.append({
            'rank':            rank,
            'movieId':         int(mid),
            'title':           info['title'],
            'genres':          info['genres'],
            'predictedRating': round(float(cache_global['media_por_item'][mid]), 4),
        })
        rank += 1
        if rank > limit:
            break
    return recomendaciones


# =============================================================================
# SECCIÓN 7 — Endpoints
# =============================================================================

# --- Modelos Pydantic ---
class LoginBody(BaseModel):
    userId: int


class NuevoRatingBody(BaseModel):
    movieId: int
    rating: float


class NuevoUsuarioBody(BaseModel):
    ratings: list[dict]


# --------------------------------------------------------------------------- #
# POST /api/login
# --------------------------------------------------------------------------- #
@app.post("/api/login")
async def login(body: LoginBody):
    """
    Autentica un usuario existente del dataset o un nuevo usuario creado por la API.
    Respuesta: { user: { userId, displayName, totalRatings, joinedAt } }
    """
    _verificar_cache()
    uid                = body.userId
    user_ratings_cache = cache_global['user_ratings']
    join_dates         = cache_global.get('user_join_dates', {})

    # Usuario del dataset original
    if uid in user_ratings_cache:
        total  = len(user_ratings_cache[uid])
        joined = join_dates.get(uid, '2000-01-01')
        return {
            "user": {
                "userId":       uid,
                "displayName":  f"Usuario {uid}",
                "totalRatings": total,
                "joinedAt":     joined,
            }
        }

    # Usuario creado vía API
    for u in cargar_nuevos_usuarios():
        if u['userId'] == uid:
            return {
                "user": {
                    "userId":       uid,
                    "displayName":  "Nuevo Usuario",
                    "totalRatings": u.get('totalRatings', 0),
                    "joinedAt":     u.get('joinedAt', ''),
                }
            }

    raise HTTPException(status_code=404, detail=f"Usuario {uid} no encontrado.")


# --------------------------------------------------------------------------- #
# POST /api/users
# --------------------------------------------------------------------------- #
@app.post("/api/users")
async def crear_usuario(body: NuevoUsuarioBody):
    """
    Registra un nuevo usuario con sus ratings iniciales.
    Respuesta: { user: { userId, displayName, totalRatings, joinedAt } }
    """
    _verificar_cache()
    nuevos    = cargar_nuevos_usuarios()
    nuevo_id  = 500001 + len(nuevos)
    hoy       = datetime.date.today().isoformat()
    total     = len(body.ratings)

    user_data = {
        'userId':       nuevo_id,
        'displayName':  'Nuevo Usuario',
        'totalRatings': total,
        'joinedAt':     hoy,
    }
    await guardar_nuevo_usuario(user_data)

    for r in body.ratings:
        await guardar_rating_extra(nuevo_id, int(r['movieId']), float(r['rating']), hoy)

    return {"user": user_data}


# --------------------------------------------------------------------------- #
# GET /api/users/{id}/ratings
# --------------------------------------------------------------------------- #
@app.get("/api/users/{user_id}/ratings")
async def obtener_ratings(user_id: int):
    """
    Devuelve el historial de ratings del usuario.
    Combina ratings del dataset (caché) con extras guardados en disco.
    Respuesta: { ratings: [{ movieId, title, genres, rating, ratedAt }] }
    """
    _verificar_cache()
    movies_dict        = cache_global['movies_dict']
    user_ratings_cache = cache_global['user_ratings']

    resultado: list = []
    vistos: set     = set()

    # Ratings extra tienen prioridad (los más recientes primero)
    for r in cargar_ratings_extra().get(user_id, []):
        mid  = int(r['movieId'])
        info = movies_dict.get(mid, {'title': f'Movie {mid}', 'genres': []})
        resultado.append({
            'movieId': mid,
            'title':   info['title'],
            'genres':  info['genres'],
            'rating':  r['rating'],
            'ratedAt': r.get('ratedAt', '1970-01-01'),
        })
        vistos.add(mid)

    # Ratings del dataset (omitir los que ya aparecen en extras)
    for r in user_ratings_cache.get(user_id, []):
        mid = int(r['movieId'])
        if mid in vistos:
            continue
        info = movies_dict.get(mid, {'title': f'Movie {mid}', 'genres': []})
        resultado.append({
            'movieId': mid,
            'title':   info['title'],
            'genres':  info['genres'],
            'rating':  r['rating'],
            'ratedAt': r.get('ratedAt', '1970-01-01'),
        })

    return {"ratings": resultado}


# --------------------------------------------------------------------------- #
# POST /api/users/{id}/ratings
# --------------------------------------------------------------------------- #
@app.post("/api/users/{user_id}/ratings")
async def valorar_pelicula(user_id: int, body: NuevoRatingBody):
    """
    Guarda un nuevo rating del usuario. Persiste en user_ratings_extra.json.
    Respuesta: { rating: { movieId, title, genres, rating, ratedAt } }
    """
    _verificar_cache()
    if not (0.5 <= body.rating <= 5.0):
        raise HTTPException(status_code=400, detail="El rating debe estar entre 0.5 y 5.0")

    hoy  = datetime.date.today().isoformat()
    await guardar_rating_extra(user_id, body.movieId, body.rating, hoy)

    info = cache_global['movies_dict'].get(
        body.movieId, {'title': f'Movie {body.movieId}', 'genres': []}
    )
    return {
        "rating": {
            "movieId": body.movieId,
            "title":   info['title'],
            "genres":  info['genres'],
            "rating":  body.rating,
            "ratedAt": hoy,
        }
    }


# --------------------------------------------------------------------------- #
# GET /api/movies/search?q=...
# --------------------------------------------------------------------------- #
@app.get("/api/movies/search")
async def buscar_peliculas(q: str = ""):
    """
    Búsqueda de películas por título (máximo 20 resultados).
    Respuesta: { movies: [{ movieId, title, genres }] }
    """
    _verificar_cache()
    if not q or len(q.strip()) < 1:
        return {"movies": []}

    q_lower    = q.lower().strip()
    movies_dict = cache_global['movies_dict']
    resultados: list = []

    for mid, info in movies_dict.items():
        if q_lower in info['title'].lower():
            resultados.append({
                'movieId': mid,
                'title':   info['title'],
                'genres':  info['genres'],
            })
            if len(resultados) >= 20:
                break

    return {"movies": resultados}


# --------------------------------------------------------------------------- #
# GET /api/users/{id}/recommendations
# --------------------------------------------------------------------------- #
@app.get("/api/users/{user_id}/recommendations")
async def obtener_recomendaciones(
    user_id: int,
    model: str         = 'item-item',
    similarity: str    = 'pearson',
    neighborMode: str  = 'k',
    k: int             = 20,
    threshold: float   = 0.3,
    significanceWeighting: bool = False,
    significanceAlpha: int      = 50,
    limit: int         = 10,
):
    """
    Genera recomendaciones para el usuario usando el modelo ítem-ítem.

    Algoritmo en dos fases:
      1. Generación vectorizada de ~100 candidatos por similitud máxima.
      2. Scoring con predecir_rating_con_ratings para cada candidato.

    Respuesta: { recommendations: [{ rank, movieId, title, genres, predictedRating }] }
    """
    _verificar_cache()
    sim_matrix   = _get_sim_matrix(similarity)
    modo_vecinos = 'top_k' if neighborMode == 'k' else 'umbral'
    movies_dict  = cache_global['movies_dict']

    # Pesos McLaughlin (se recalculan si gamma ≠ 25; operación vectorizada rápida)
    pesos = None
    if significanceWeighting:
        if significanceAlpha == 25:
            pesos = cache_global['pesos_25']
        else:
            pesos = calcular_pesos_mclaughlin(
                cache_global['matriz_co_ratings'], gamma=significanceAlpha
            )

    # Construir ratings del usuario (caché + extras)
    ratings_row    = _construir_ratings_row(user_id)
    all_rated_ids  = set(int(m) for m in ratings_row.dropna().index)

    # Fallback si el usuario no tiene historial
    if not all_rated_ids:
        return {"recommendations": _fallback_populares(limit, all_rated_ids, movies_dict)}

    # --- Fase 1: generación de candidatos (vectorizada) ---
    rated_en_sim = [m for m in all_rated_ids if m in sim_matrix.index]
    if not rated_en_sim:
        return {"recommendations": _fallback_populares(limit, all_rated_ids, movies_dict)}

    submatrix   = sim_matrix.loc[rated_en_sim]
    max_sim     = submatrix.max(axis=0)
    candidatos  = max_sim.drop(labels=list(all_rated_ids), errors='ignore')
    top_candidatos = candidatos.nlargest(100).index.tolist()

    # --- Fase 2: scoring de los ~100 candidatos ---
    scores = []
    for mid in top_candidatos:
        pred = predecir_rating_con_ratings(
            mid, ratings_row, sim_matrix,
            modo_vecinos=modo_vecinos,
            vecinos_k=k,
            umbral_similitud=threshold,
            pesos_significancia=pesos,
        )
        scores.append((int(mid), pred))

    scores.sort(key=lambda x: -x[1])

    recomendaciones = []
    for rank, (mid, pred) in enumerate(scores[:limit], start=1):
        info = movies_dict.get(mid, {'title': f'Movie {mid}', 'genres': []})
        recomendaciones.append({
            'rank':            rank,
            'movieId':         mid,
            'title':           info['title'],
            'genres':          info['genres'],
            'predictedRating': round(pred, 4),
        })

    return {"recommendations": recomendaciones}


# --------------------------------------------------------------------------- #
# GET /api/users/{id}/recommendations/{movieId}/explain
# --------------------------------------------------------------------------- #
@app.get("/api/users/{user_id}/recommendations/{movie_id}/explain")
async def explicar_recomendacion(
    user_id: int,
    movie_id: int,
    similarity: str    = 'pearson',
    neighborMode: str  = 'k',
    k: int             = 20,
    threshold: float   = 0.3,
    significanceWeighting: bool = False,
    significanceAlpha: int      = 50,
):
    """
    Explica por qué se recomienda movie_id al usuario:
    muestra los ítems del historial del usuario más similares a la película recomendada.

    Respuesta: { explanation: { ..., userRatingsEvidence, neighborUsers, neighborItems } }
    """
    _verificar_cache()
    sim_matrix   = _get_sim_matrix(similarity)
    modo_vecinos = 'top_k' if neighborMode == 'k' else 'umbral'
    movies_dict  = cache_global['movies_dict']

    pesos = None
    if significanceWeighting:
        if significanceAlpha == 25:
            pesos = cache_global['pesos_25']
        else:
            pesos = calcular_pesos_mclaughlin(
                cache_global['matriz_co_ratings'], gamma=significanceAlpha
            )

    ratings_row = _construir_ratings_row(user_id)

    # Predicción con detalle de vecinos usados
    resultado  = predecir_rating_explicado(
        movie_id, ratings_row, sim_matrix,
        modo_vecinos=modo_vecinos,
        vecinos_k=k,
        umbral_similitud=threshold,
        pesos_significancia=pesos,
    )
    prediccion = resultado['prediccion']
    vecinos    = resultado['vecinos']  # [(id_vecino, sim, rating_u, media_vecino), ...]

    info_objetivo = movies_dict.get(movie_id, {'title': f'Movie {movie_id}', 'genres': []})
    movie_avg     = float(cache_global['media_por_item'].get(movie_id, rating_promedio_global))

    # Construir evidencia (vecinos ordenados por similitud descendente)
    vecinos_ordenados = sorted(vecinos, key=lambda x: -x[1])

    user_ratings_evidence = []
    neighbor_items        = []

    for (id_vecino, sim, rating_u, media_vecino) in vecinos_ordenados:
        info_v = movies_dict.get(id_vecino, {'title': f'Movie {id_vecino}', 'genres': []})
        user_ratings_evidence.append({
            'movieId':    id_vecino,
            'title':      info_v['title'],
            'genres':     info_v['genres'],
            'rating':     round(rating_u, 2),
            'similarity': round(sim, 4),
        })
        neighbor_items.append({
            'movieId':    id_vecino,
            'title':      info_v['title'],
            'avgRating':  round(media_vecino, 4),
            'similarity': round(sim, 4),
        })

    # Descripción del modelo usado
    config_str = f"k={k}" if neighborMode == 'k' else f"umbral={threshold}"
    model_used = f"Item-Item — {similarity.capitalize()}, {config_str}"
    if significanceWeighting:
        model_used += f", McLaughlin γ={significanceAlpha}"

    return {
        "explanation": {
            "userId":               user_id,
            "movieId":              movie_id,
            "movieTitle":           info_objetivo['title'],
            "movieGenres":          info_objetivo['genres'],
            "movieAvgRating":       round(movie_avg, 4),
            "predictedRating":      round(prediccion, 4),
            "modelUsed":            model_used,
            "similarityMetric":     similarity,
            "neighborsUsed":        len(vecinos),
            "userRatingsEvidence":  user_ratings_evidence,
            "neighborUsers":        [],
            "neighborItems":        neighbor_items,
        }
    }


# =============================================================================
# SECCIÓN 8 — Punto de entrada principal
# =============================================================================

if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else 'build'

    if modo == 'build':
        print("=" * 64)
        print("MODO BUILD — Construyendo caché ítem-ítem")
        print("=" * 64)

        cache = construir_cache()

        # --- Resumen final ---
        n_u, n_i = cache['matriz_usuario_item'].shape
        densidad = (
            cache['matriz_usuario_item'].notna().sum().sum() / (n_u * n_i) * 100
        )
        print("\n" + "=" * 64)
        print("RESUMEN DEL MODELO")
        print(f"  Usuarios : {n_u:,}")
        print(f"  Películas: {n_i:,}")
        print(f"  Densidad : {densidad:.2f}%")
        print()
        for nombre, sim_key in [
            ('Coseno',  'sim_coseno'),
            ('Pearson', 'sim_pearson'),
            ('Jaccard', 'sim_jaccard'),
        ]:
            res = evaluar_modelo(
                cache['test_set'],
                cache['matriz_usuario_item'],
                cache[sim_key],
            )
            print(
                f"  RMSE {nombre}: {res['rmse']:.4f} | "
                f"MAE {nombre}: {res['mae']:.4f}  "
                f"(n={res['n_evaluados']})"
            )
        print("=" * 64)

    elif modo == 'serve':
        import uvicorn
        uvicorn.run("Item-Item:app", host="0.0.0.0", port=8000, reload=False)

    else:
        print(f"Modo desconocido: '{modo}'. Opciones: build | serve")
        sys.exit(1)
