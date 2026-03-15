"""
main.py — Servidor FastAPI con el mejor modelo user-user.
    uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, pickle, os
import pandas as pd
from surprise import Reader, Dataset
app = FastAPI(title="MovieLens Recommender")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class S:
    trainset       = None
    movies_dict    = {}
    user_ratings   = {}
    user_neighbors = {}   # vecinos precalculados {userId: [{userId, similarity}]}
    algo           = None
    best_params    = {}
s = S()

@app.on_event("startup")
def startup():
    if not os.path.exists("model_cache.pkl"):
        raise RuntimeError("Corre primero la celda 8 del notebook")
    print("Cargando model_cache.pkl...")
    cache = pickle.load(open("model_cache.pkl", "rb"))
    s.trainset       = cache["trainset"]
    s.movies_dict    = cache["movies_dict"]
    s.user_ratings   = {int(k): v for k, v in cache["user_ratings"].items()}
    s.user_neighbors = {int(k): v for k, v in cache.get("user_neighbors", {}).items()}
    s.algo           = cache["best_model"]
    s.best_params    = cache.get("best_params", {})
    print(f"  {len(s.user_ratings):,} usuarios | modelo: {s.best_params}")

    # Cargar ratings extras guardados en disco
    if os.path.exists("user_ratings_extra.json"):
        import json
        with open("user_ratings_extra.json") as f:
            extra = json.load(f)
        for uid, ratings in extra.items():
            s.user_ratings[int(uid)] = ratings
        print(f"Ratings extra cargados: {len(extra)} usuarios")

def _movie(mid):
    mid = int(mid)
    i = s.movies_dict.get(mid, {})
    return {"movieId": mid, "title": i.get("title", f"Movie {mid}"), "genres": i.get("genres", [])}

class RateBody(BaseModel):
    movieId: int
    rating:  float

class NewUserBody(BaseModel):
    ratings: list

# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "users":  len(s.user_ratings),
        "params": s.best_params,
        "model_info": {
            "type":       "user-user",
            "similarity": "pearson_baseline",
            "k":          s.best_params.get("k", 50),
            "gamma":      s.best_params.get("gamma", 200),
            "mae":        s.best_params.get("mae", 0.5880),
            "rmse":       s.best_params.get("rmse", 0.7718),
        }
    }

# ── Login ────────────────────────────────────────────────────────────────────
@app.get("/api/users/{user_id}")
def get_user(user_id: int):
    if user_id not in s.user_ratings:
        valid = sorted(s.user_ratings.keys())[:10]
        raise HTTPException(404, detail=f"Usuario {user_id} no encontrado. Algunos IDs válidos: {valid}")
    return {"user": {
        "userId":       user_id,
        "displayName":  f"Usuario {user_id}",
        "totalRatings": len(s.user_ratings[user_id]),
        "joinedAt":     "2000-01-01",
    }}

# ── Crear nuevo usuario ──────────────────────────────────────────────────────
@app.post("/api/users")
def create_user(body: NewUserBody):
    new_id = max(s.user_ratings.keys()) + 1 if s.user_ratings else 999999
    s.user_ratings[new_id] = [
        {"movieId": int(r["movieId"]), "rating": float(r["rating"])}
        for r in body.ratings
    ]
    
    # Integrar al modelo con los ratings iniciales
    rows = [(s.trainset.to_raw_uid(u), s.trainset.to_raw_iid(i), r)
            for u, i, r in s.trainset.all_ratings()]
    for r in s.user_ratings[new_id]:
        if s.trainset.knows_item(r["movieId"]):
            rows.append((new_id, r["movieId"], r["rating"]))

    df           = pd.DataFrame(rows, columns=["userId","movieId","rating"])
    reader       = Reader(rating_scale=(0.5, 5.0))
    dataset      = Dataset.load_from_df(df[["userId","movieId","rating"]], reader)
    new_trainset = dataset.build_full_trainset()
    s.algo.fit(new_trainset)
    s.trainset = new_trainset

    return {"user": {
        "userId":       new_id,
        "displayName":  f"Usuario {new_id}",
        "totalRatings": len(body.ratings),
        "joinedAt":     "2024-01-01",
    }}

# ── Historial de ratings ─────────────────────────────────────────────────────
@app.get("/api/users/{user_id}/ratings")
def get_ratings(user_id: int):
    if user_id not in s.user_ratings:
        raise HTTPException(404, "Usuario no encontrado")
    return {"ratings": [
        {**_movie(r["movieId"]), "rating": r["rating"], "ratedAt": "2000-01-01"}
        for r in s.user_ratings[user_id][:50]
    ]}

# ── Guardar rating ───────────────────────────────────────────────────────────
@app.post("/api/users/{user_id}/ratings")
def rate_movie(user_id: int, body: RateBody):
    if user_id not in s.user_ratings:
        s.user_ratings[user_id] = []
    s.user_ratings[user_id] = [r for r in s.user_ratings[user_id] if r["movieId"] != body.movieId]
    s.user_ratings[user_id].insert(0, {"movieId": body.movieId, "rating": body.rating})
    
    # Guardar en disco para persistir entre reinicios
    import json
    with open("user_ratings_extra.json", "w") as f:
        json.dump(s.user_ratings, f)
    
    return {"rating": {**_movie(body.movieId), "rating": body.rating, "ratedAt": "2024-01-01"}}

# ── Recomendaciones ──────────────────────────────────────────────────────────
@app.get("/api/users/{user_id}/recommendations")
def get_recs(user_id: int, limit: int = Query(10)):
    if not s.trainset.knows_user(user_id):
        # Usuario nuevo → películas más populares
        pop = sorted(s.movies_dict.items(), key=lambda x: x[0])[:limit]
        return {"recommendations": [
            {**_movie(mid), "rank": i+1, "predictedRating": 4.0}
            for i, (mid, _) in enumerate(pop)
        ]}

    rated = {r["movieId"] for r in s.user_ratings.get(user_id, [])}
    preds = []
    for inner_iid in s.trainset.all_items():
        raw_iid = s.trainset.to_raw_iid(inner_iid)
        if raw_iid in rated:
            continue
        pred = s.algo.predict(user_id, raw_iid)
        if not pred.details.get("was_impossible", False):
            preds.append((raw_iid, pred.est))

    preds.sort(key=lambda x: x[1], reverse=True)
    return {"recommendations": [
        {**_movie(mid), "rank": i+1, "predictedRating": round(r, 2)}
        for i, (mid, r) in enumerate(preds[:limit])
    ]}

# ── Explicación ──────────────────────────────────────────────────────────────
@app.get("/api/users/{user_id}/recommendations/{movie_id}/explain")
def explain(user_id: int, movie_id: int):
    try:
        inner_uid = s.trainset.to_inner_uid(user_id)
        inner_iid = s.trainset.to_inner_iid(movie_id)
    except ValueError:
        raise HTTPException(404, "Usuario o película no encontrados en el modelo")

    pred       = s.algo.predict(user_id, movie_id)
    user_items = {j: r for j, r in s.trainset.ur[inner_uid]}

    # Usar vecinos precalculados
    precomputed = s.user_neighbors.get(user_id, [])

    neighbors_out = []
    for nb in precomputed:
        nb_uid = nb["userId"]
        sim_val = nb["similarity"]

        try:
            nb_inner = s.trainset.to_inner_uid(nb_uid)
        except ValueError:
            continue

        # ¿Calificó esta película?
        nb_rating = next((r for j, r in s.trainset.ur[nb_inner] if j == inner_iid), None)
        if nb_rating is None:
            continue

        # Películas en común (muestra las primeras 5)
        nb_items = {j: r for j, r in s.trainset.ur[nb_inner]}
        shared   = []
        for j in list(set(user_items) & set(nb_items))[:5]:
            raw_j = s.trainset.to_raw_iid(j)
            shared.append({
                **_movie(raw_j),
                "userRating":     round(user_items[j], 1),
                "neighborRating": round(nb_items[j], 1),
            })

        neighbors_out.append({
            "userId":         nb_uid,
            "displayName":    f"Usuario {nb_uid}",
            "similarity":     sim_val,
            "ratingForMovie": round(nb_rating, 2),
            "sharedMovies":   shared,
        })

        if len(neighbors_out) >= 8:
            break

    # Info global de la película
    movie_ratings = [r for _, r in s.trainset.ir[inner_iid]]
    avg_rating    = round(float(np.mean(movie_ratings)), 2) if movie_ratings else 0.0
    info          = _movie(movie_id)
    params        = s.best_params

    return {"explanation": {
        "userId":          user_id,
        "movieId":         movie_id,
        "movieTitle":      info["title"],
        "movieGenres":     info["genres"],
        "movieAvgRating":  avg_rating,
        "predictedRating": round(pred.est, 2),
        "modelUsed":       f"User-User — pearson_baseline, k={params.get('k',50)}, γ={params.get('gamma',200)}",
        "neighborUsers":   neighbors_out,   # vecinos que calificaron esta película
        "neighborItems":   [],
        "userRatingsEvidence": [
            {**_movie(s.trainset.to_raw_iid(j)), "rating": round(r, 1), "similarity": 0.5}
            for j, r in list(user_items.items())[:10]
        ],
    }}

# ── Búsqueda ─────────────────────────────────────────────────────────────────
@app.get("/api/movies/search")
def search(q: str = Query("")):
    if not q.strip(): return {"movies": []}
    q_l = q.lower()
    return {"movies": [
        {"movieId": mid, **info}
        for mid, info in s.movies_dict.items()
        if q_l in info.get("title", "").lower()
    ][:20]}