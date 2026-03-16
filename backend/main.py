"""
main.py — Servidor unificado User-User + Item-Item
    python -m uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, pickle, os, json
import pandas as pd
from surprise import Reader, Dataset
from typing import Optional

app = FastAPI(title="MovieLens Recommender")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# =============================================================================
# Estado global
# =============================================================================
class S:
    uu_trainset  = None
    uu_algo      = None
    uu_neighbors = {}
    uu_params    = {}
    uu_loaded    = False
    ii_cache         = None
    ii_media_items: Optional[pd.Series] = None
    ii_rating_global: Optional[float]   = None
    ii_loaded    = False
    movies_dict  = {}
    user_ratings = {}
s = S()

# Usuarios y ratings que vienen de los pickles originales
_original_users: set   = set()
_original_movie_ids: dict = {}   # {userId: set de movieIds originales}

# =============================================================================
# Startup
# =============================================================================
@app.on_event("startup")
def startup():
    # ── User-User ──────────────────────────────────────────────────────────
    if os.path.exists("model_cache.pkl"):
        print("Cargando model_cache.pkl (User-User)...")
        uu = pickle.load(open("model_cache.pkl", "rb"))
        s.uu_trainset  = uu["trainset"]
        s.uu_algo      = uu["best_model"]
        s.uu_neighbors = {int(k): v for k, v in uu.get("user_neighbors", {}).items()}
        s.uu_params    = uu.get("best_params", {})
        s.movies_dict.update(uu["movies_dict"])
        for uid, rats in uu["user_ratings"].items():
            s.user_ratings[int(uid)] = list(rats)
        s.uu_loaded = True
        print(f"  User-User OK — {len(uu['user_ratings']):,} usuarios")
    else:
        print("AVISO: model_cache.pkl no encontrado")

    # ── Item-Item ──────────────────────────────────────────────────────────
    if os.path.exists("item_item_cache.pkl"):
        print("Cargando item_item_cache.pkl (Item-Item)...")
        ii = pickle.load(open("item_item_cache.pkl", "rb"))
        s.ii_cache         = ii
        s.ii_media_items   = ii["media_por_item"]
        s.ii_rating_global = float(ii["rating_promedio_global"])
        if ii.get("movies_dict"):
            for mid, info in ii["movies_dict"].items():
                if mid not in s.movies_dict:
                    s.movies_dict[mid] = info
        for uid, rats in ii.get("user_ratings", {}).items():
            if int(uid) not in s.user_ratings:
                s.user_ratings[int(uid)] = list(rats)
        s.ii_loaded = True
        print(f"  Item-Item OK — {len(ii.get('user_ratings',{})):,} usuarios")
    else:
        print("AVISO: item_item_cache.pkl no encontrado")

    # Marcar usuarios y sus movieIds originales ANTES de cargar extras
    _original_users.update(s.user_ratings.keys())
    for uid, rats in s.user_ratings.items():
        _original_movie_ids[uid] = {r["movieId"] for r in rats}

    # ── Ratings extra guardados en sesiones anteriores ─────────────────────
    if os.path.exists("user_ratings_extra.json"):
        with open("user_ratings_extra.json") as f:
            extra = json.load(f)
        for uid, rats in extra.items():
            s.user_ratings[int(uid)] = rats
        print(f"  Ratings extra restaurados: {len(extra)} usuarios")

    print(f"Servidor listo. Usuarios: {len(s.user_ratings):,}")


# =============================================================================
# Helpers generales
# =============================================================================
def _movie(mid):
    mid = int(mid)
    i = s.movies_dict.get(mid, {})
    return {"movieId": mid, "title": i.get("title", f"Movie {mid}"), "genres": i.get("genres", [])}

def _save_extra():
    """
    Guarda en disco:
    - Usuarios nuevos (no estaban en los pickles)
    - Usuarios existentes que calificaron películas nuevas
    """
    extra = {}
    for uid, rats in s.user_ratings.items():
        if uid not in _original_users:
            # Usuario nuevo — guardar todo
            extra[str(uid)] = rats
        else:
            # Usuario existente — guardar solo si tiene ratings nuevos
            orig_ids = _original_movie_ids.get(uid, set())
            nuevos   = [r for r in rats if r["movieId"] not in orig_ids]
            if nuevos:
                extra[str(uid)] = rats
    with open("user_ratings_extra.json", "w") as f:
        json.dump(extra, f)

# =============================================================================
# Helpers Item-Item
# =============================================================================
def _ii_sim(similarity: str) -> pd.DataFrame:
    mapa = {"cosine":  s.ii_cache["sim_coseno"],
            "pearson": s.ii_cache["sim_pearson"],
            "jaccard": s.ii_cache["sim_jaccard"]}
    return mapa.get(similarity, s.ii_cache["sim_pearson"])

def _ii_ratings_row(user_id: int) -> pd.Series:
    matriz = s.ii_cache["matriz_usuario_item"]
    row    = matriz.loc[user_id].copy() if user_id in matriz.index else pd.Series(dtype=float)
    for r in s.user_ratings.get(user_id, []):
        row[int(r["movieId"])] = r["rating"]
    return row

def _ii_predict(movie_id, ratings_row, sim_matrix, k=20):
    if movie_id not in sim_matrix.index:
        return s.ii_rating_global
    rated   = ratings_row.dropna().drop(movie_id, errors="ignore")
    if rated.empty: return s.ii_rating_global
    comunes = rated.index.intersection(sim_matrix.columns)
    if comunes.empty: return s.ii_rating_global
    sims    = sim_matrix.loc[movie_id, comunes]
    vecinos = sims[sims > 0].nlargest(k)
    if vecinos.empty: return s.ii_rating_global
    media_obj = float(s.ii_media_items.get(movie_id, s.ii_rating_global))
    num = sum(sim * (float(rated[nb]) - float(s.ii_media_items.get(nb, s.ii_rating_global)))
              for nb, sim in vecinos.items())
    den = vecinos.abs().sum()
    if den == 0: return s.ii_rating_global
    return float(np.clip(media_obj + num / den, 0.5, 5.0))

def _ii_predict_explained(movie_id, ratings_row, sim_matrix, k=20):
    if movie_id not in sim_matrix.index:
        return s.ii_rating_global, []
    rated   = ratings_row.dropna().drop(movie_id, errors="ignore")
    if rated.empty: return s.ii_rating_global, []
    comunes = rated.index.intersection(sim_matrix.columns)
    if comunes.empty: return s.ii_rating_global, []
    sims    = sim_matrix.loc[movie_id, comunes]
    vecinos = sims[sims > 0].nlargest(k)
    if vecinos.empty: return s.ii_rating_global, []
    media_obj       = float(s.ii_media_items.get(movie_id, s.ii_rating_global))
    num, den, det   = 0.0, 0.0, []
    for nb, sim in vecinos.items():
        mb  = float(s.ii_media_items.get(nb, s.ii_rating_global))
        r_u = float(rated[nb])
        num += sim * (r_u - mb)
        den += abs(sim)
        det.append((int(nb), float(sim), r_u, mb))
    if den == 0: return s.ii_rating_global, det
    return float(np.clip(media_obj + num / den, 0.5, 5.0)), det

# =============================================================================
# Pydantic
# =============================================================================
class RateBody(BaseModel):
    movieId: int
    rating:  float

class NewUserBody(BaseModel):
    ratings: list

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/api/health")
def health():
    return {
        "status":    "ok",
        "users":     len(s.user_ratings),
        "user_user": "loaded" if s.uu_loaded else "not loaded",
        "item_item": "loaded" if s.ii_loaded else "not loaded",
        "uu_params": s.uu_params,
    }

# ── Login ─────────────────────────────────────────────────────────────────────
@app.get("/api/users/{user_id}")
def get_user(user_id: int):
    if user_id not in s.user_ratings:
        valid = sorted(s.user_ratings.keys())[:10]
        raise HTTPException(404, detail=f"Usuario {user_id} no encontrado. IDs validos: {valid}")
    rats = s.user_ratings[user_id]
    return {"user": {
        "userId":       user_id,
        "displayName":  f"Usuario {user_id}",
        "totalRatings": len(rats),
        "joinedAt":     rats[0].get("ratedAt", "2000-01-01") if rats else "2000-01-01",
    }}

# ── Crear nuevo usuario ───────────────────────────────────────────────────────
@app.post("/api/users")
def create_user(body: NewUserBody):
    new_id = max(s.user_ratings.keys()) + 1 if s.user_ratings else 999999
    s.user_ratings[new_id] = [
        {"movieId": int(r["movieId"]), "rating": float(r["rating"]), "ratedAt": "2024-01-01"}
        for r in body.ratings
    ]

    # Integrar al modelo User-User
    if s.uu_loaded:
        rows = [(s.uu_trainset.to_raw_uid(u), s.uu_trainset.to_raw_iid(i), r)
                for u, i, r in s.uu_trainset.all_ratings()]
        for r in s.user_ratings[new_id]:
            if s.uu_trainset.knows_item(r["movieId"]):
                rows.append((new_id, r["movieId"], r["rating"]))
        df = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
        ds = Dataset.load_from_df(df[["userId", "movieId", "rating"]], Reader(rating_scale=(0.5, 5.0)))
        ts = ds.build_full_trainset()
        s.uu_algo.fit(ts)
        s.uu_trainset = ts

        # Calcular vecinos del nuevo usuario para el modal "Por que?"
        if ts.knows_user(new_id):
            inner   = ts.to_inner_uid(new_id)
            sim_row = s.uu_algo.sim[inner]
            top_idx = np.argsort(sim_row)[::-1][1:51]
            s.uu_neighbors[new_id] = [
                {"userId": ts.to_raw_uid(nb), "similarity": round(float(sim_row[nb]), 4)}
                for nb in top_idx if sim_row[nb] > 0
            ]

    _save_extra()
    return {"user": {
        "userId":       new_id,
        "displayName":  f"Usuario {new_id}",
        "totalRatings": len(body.ratings),
        "joinedAt":     "2024-01-01",
    }}

# ── Historial ─────────────────────────────────────────────────────────────────
@app.get("/api/users/{user_id}/ratings")
def get_ratings(user_id: int):
    if user_id not in s.user_ratings:
        raise HTTPException(404, "Usuario no encontrado")
    return {"ratings": [
        {**_movie(r["movieId"]), "rating": r["rating"], "ratedAt": r.get("ratedAt", "2000-01-01")}
        for r in s.user_ratings[user_id][:50]
    ]}

# ── Guardar rating ────────────────────────────────────────────────────────────
@app.post("/api/users/{user_id}/ratings")
def rate_movie(user_id: int, body: RateBody):
    if user_id not in s.user_ratings:
        s.user_ratings[user_id] = []
    s.user_ratings[user_id] = [r for r in s.user_ratings[user_id] if r["movieId"] != body.movieId]
    s.user_ratings[user_id].insert(0, {
        "movieId": body.movieId,
        "rating":  body.rating,
        "ratedAt": "2024-01-01",
    })
    _save_extra()
    return {"rating": {**_movie(body.movieId), "rating": body.rating, "ratedAt": "2024-01-01"}}

# ── Refresh User-User ─────────────────────────────────────────────────────────
@app.post("/api/users/{user_id}/refresh")
def refresh_user(user_id: int):
    if not s.uu_loaded:
        raise HTTPException(503, "User-User no disponible")
    if user_id not in s.user_ratings:
        raise HTTPException(404, "Usuario no encontrado")
    rows = [(s.uu_trainset.to_raw_uid(u), s.uu_trainset.to_raw_iid(i), r)
            for u, i, r in s.uu_trainset.all_ratings()]
    rated_ids = {r["movieId"] for r in s.user_ratings[user_id]}
    rows = [(u, i, r) for u, i, r in rows if not (u == user_id and i in rated_ids)]
    for r in s.user_ratings[user_id]:
        rows.append((user_id, r["movieId"], r["rating"]))
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    ds = Dataset.load_from_df(df[["userId", "movieId", "rating"]], Reader(rating_scale=(0.5, 5.0)))
    ts = ds.build_full_trainset()
    s.uu_algo.fit(ts)
    s.uu_trainset = ts
    return {"status": "ok"}

# ── Recomendaciones ───────────────────────────────────────────────────────────
@app.get("/api/users/{user_id}/recommendations")
def get_recs(
    user_id:    int,
    limit:      int = Query(10),
    model:      str = Query("user-user"),
    similarity: str = Query("pearson"),
    k:          int = Query(20),
):
    rated = {r["movieId"] for r in s.user_ratings.get(user_id, [])}

    # ── Item-Item ──────────────────────────────────────────────────────────
    if model == "item-item":
        if not s.ii_loaded:
            raise HTTPException(503, "Item-Item no disponible")
        sim_matrix  = _ii_sim(similarity)
        ratings_row = _ii_ratings_row(user_id)
        all_rated   = set(int(m) for m in ratings_row.dropna().index)
        if not all_rated:
            pop = s.ii_media_items.nlargest(limit * 3).index
            return {"recommendations": [
                {**_movie(int(mid)), "rank": i+1, "predictedRating": round(float(s.ii_media_items[mid]), 2)}
                for i, mid in enumerate(p for p in pop if int(p) not in rated)
            ][:limit]}
        rated_en_sim = [m for m in all_rated if m in sim_matrix.index]
        if not rated_en_sim:
            return {"recommendations": []}
        max_sim    = sim_matrix.loc[rated_en_sim].max(axis=0)
        candidatos = max_sim.drop(labels=list(all_rated), errors="ignore").nlargest(100).index
        scores     = [(int(mid), _ii_predict(int(mid), ratings_row, sim_matrix, k=k))
                      for mid in candidatos]
        scores.sort(key=lambda x: -x[1])
        return {"recommendations": [
            {**_movie(mid), "rank": i+1, "predictedRating": round(r, 2)}
            for i, (mid, r) in enumerate(scores[:limit])
        ]}

    # ── User-User ──────────────────────────────────────────────────────────
    if not s.uu_loaded:
        raise HTTPException(503, "User-User no disponible")
    if not s.uu_trainset.knows_user(user_id):
        # Usuario no esta en el modelo → fallback peliculas populares por rating promedio
        if s.ii_loaded:
            pop = s.ii_media_items.nlargest(limit * 3).index
            return {"recommendations": [
                {**_movie(int(mid)), "rank": i+1, "predictedRating": round(float(s.ii_media_items[mid]), 2)}
                for i, mid in enumerate(p for p in pop if int(p) not in rated)
            ][:limit]}
        pop = sorted(s.movies_dict.items(), key=lambda x: x[0])[:limit*2]
        return {"recommendations": [
            {**_movie(mid), "rank": i+1, "predictedRating": 4.0}
            for i, (mid, _) in enumerate(p for p in pop if p[0] not in rated)
        ][:limit]}

    preds = []
    for inner_iid in s.uu_trainset.all_items():
        raw_iid = s.uu_trainset.to_raw_iid(inner_iid)
        if raw_iid in rated:
            continue
        pred = s.uu_algo.predict(user_id, raw_iid)
        if not pred.details.get("was_impossible", False):
            preds.append((raw_iid, pred.est))
    preds.sort(key=lambda x: x[1], reverse=True)
    return {"recommendations": [
        {**_movie(mid), "rank": i+1, "predictedRating": round(r, 2)}
        for i, (mid, r) in enumerate(preds[:limit])
    ]}

# ── Explicacion ───────────────────────────────────────────────────────────────
@app.get("/api/users/{user_id}/recommendations/{movie_id}/explain")
def explain(
    user_id:    int,
    movie_id:   int,
    model:      str = Query("user-user"),
    similarity: str = Query("pearson"),
    k:          int = Query(20),
):
    # ── Item-Item ──────────────────────────────────────────────────────────
    if model == "item-item":
        if not s.ii_loaded:
            raise HTTPException(503, "Item-Item no disponible")
        sim_matrix    = _ii_sim(similarity)
        ratings_row   = _ii_ratings_row(user_id)
        pred, detalle = _ii_predict_explained(movie_id, ratings_row, sim_matrix, k=k)
        info = _movie(movie_id)
        avg  = float(s.ii_media_items.get(movie_id, s.ii_rating_global))
        det  = sorted(detalle, key=lambda x: -x[1])
        return {"explanation": {
            "userId":          user_id,
            "movieId":         movie_id,
            "movieTitle":      info["title"],
            "movieGenres":     info["genres"],
            "movieAvgRating":  round(avg, 2),
            "predictedRating": round(pred, 2),
            "modelUsed":       f"Item-Item — {similarity}, k={k}",
            "neighborUsers":   [],
            "neighborItems":   [
                {**_movie(nb), "avgRating": round(m, 2), "similarity": round(sim, 4)}
                for nb, sim, r_u, m in det
            ],
            "userRatingsEvidence": [
                {**_movie(nb), "rating": round(r_u, 1), "similarity": round(sim, 4)}
                for nb, sim, r_u, m in det
            ],
        }}

    # ── User-User ──────────────────────────────────────────────────────────
    if not s.uu_loaded:
        raise HTTPException(503, "User-User no disponible")
    try:
        inner_uid = s.uu_trainset.to_inner_uid(user_id)
        inner_iid = s.uu_trainset.to_inner_iid(movie_id)
    except ValueError:
        raise HTTPException(404, "Usuario o pelicula no encontrados en el modelo")

    pred       = s.uu_algo.predict(user_id, movie_id)
    user_items = {j: r for j, r in s.uu_trainset.ur[inner_uid]}
    precomp    = s.uu_neighbors.get(user_id, [])

    neighbors_out = []
    for nb in precomp:
        nb_uid, sim_val = nb["userId"], nb["similarity"]
        try:
            nb_inner = s.uu_trainset.to_inner_uid(nb_uid)
        except ValueError:
            continue
        nb_rating = next((r for j, r in s.uu_trainset.ur[nb_inner] if j == inner_iid), None)
        if nb_rating is None:
            continue
        nb_items = {j: r for j, r in s.uu_trainset.ur[nb_inner]}
        shared   = [
            {**_movie(s.uu_trainset.to_raw_iid(j)),
             "userRating": round(user_items[j], 1),
             "neighborRating": round(nb_items[j], 1)}
            for j in list(set(user_items) & set(nb_items))[:5]
        ]
        neighbors_out.append({
            "userId":         nb_uid,
            "displayName":    f"Usuario {nb_uid}",
            "similarity":     sim_val,
            "ratingForMovie": round(nb_rating, 2),
            "sharedMovies":   shared,
        })
        if len(neighbors_out) >= 8:
            break

    movie_ratings = [r for _, r in s.uu_trainset.ir[inner_iid]]
    avg  = round(float(np.mean(movie_ratings)), 2) if movie_ratings else 0.0
    info = _movie(movie_id)
    p    = s.uu_params

    return {"explanation": {
        "userId":          user_id,
        "movieId":         movie_id,
        "movieTitle":      info["title"],
        "movieGenres":     info["genres"],
        "movieAvgRating":  avg,
        "predictedRating": round(pred.est, 2),
        "modelUsed":       f"User-User — pearson_baseline, k={p.get('k',50)}, gamma={p.get('gamma',200)}",
        "neighborUsers":   neighbors_out,
        "neighborItems":   [],
        "userRatingsEvidence": [
            {**_movie(s.uu_trainset.to_raw_iid(j)), "rating": round(r, 1), "similarity": 0.5}
            for j, r in list(user_items.items())[:10]
        ],
    }}

# ── Busqueda ──────────────────────────────────────────────────────────────────
@app.get("/api/movies/search")
def search(q: str = Query("")):
    if not q.strip():
        return {"movies": []}
    q_l = q.lower()
    return {"movies": [
        {"movieId": mid, **info}
        for mid, info in s.movies_dict.items()
        if q_l in info.get("title", "").lower()
    ][:20]}