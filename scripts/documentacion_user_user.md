# Sistema de Recomendación Usuario-Usuario — Documentación Completa

> **Archivo:** `backend/user_user.py`
> **Dataset:** MovieLens 20M
> **Stack:** Python · NumPy · Pandas · scikit-surprise · FastAPI · React

---

## Tabla de Contenidos

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Arquitectura general](#2-arquitectura-general)
3. [El dataset](#3-el-dataset)
4. [Pipeline de datos](#4-pipeline-de-datos)
5. [La métrica de similitud](#5-la-métrica-de-similitud)
6. [Pesos de significancia McLaughlin](#6-pesos-de-significancia-mclaughlin)
7. [El modelo de predicción](#7-el-modelo-de-predicción)
8. [Algoritmo de recomendaciones](#8-algoritmo-de-recomendaciones)
9. [Algoritmo de explicación](#9-algoritmo-de-explicación)
10. [Estructura del caché](#10-estructura-del-caché)
11. [Persistencia en tiempo de ejecución](#11-persistencia-en-tiempo-de-ejecución)
12. [Endpoints de la API](#12-endpoints-de-la-api)
13. [Flujo completo end-to-end](#13-flujo-completo-end-to-end)
14. [Cómo correr el sistema](#14-cómo-correr-el-sistema)
15. [Diferencias con el modelo Ítem-Ítem](#15-diferencias-con-el-modelo-ítem-ítem)
16. [Limitaciones y consideraciones](#16-limitaciones-y-consideraciones)

---

## 1. Resumen ejecutivo

El sistema implementa **filtrado colaborativo usuario-usuario** sobre el dataset MovieLens 20M. Dado un usuario activo, encuentra los usuarios más similares a él (vecinos) en base a sus patrones de valoración históricos, y predice el rating que daría a películas que no ha visto usando un promedio ponderado de los ratings de esos vecinos.

El resultado es un servidor FastAPI (`main.py`) que expone endpoints REST consumidos por una aplicación React. El modelo utiliza la métrica **Pearson Baseline con significance weighting McLaughlin** como mejor configuración según los experimentos realizados (MAE=0.588, RMSE=0.771).

---

## 2. Arquitectura general

```
┌─────────────────────────────────────────┐
│           FRONTEND (React :3000)         │
│  LoginPage │ CreateUserPage │ Dashboard  │
│            src/api/api.js               │
└──────────────────┬──────────────────────┘
                   │  HTTP (fetch)
                   ▼
┌─────────────────────────────────────────┐
│      BACKEND FastAPI (:8000)  main.py   │
│  GET  /api/health                       │
│  GET  /api/users/:id                    │
│  POST /api/users                        │
│  GET  /api/users/:id/ratings            │
│  POST /api/users/:id/ratings            │
│  POST /api/users/:id/refresh            │
│  GET  /api/movies/search                │
│  GET  /api/users/:id/recommendations    │
│  GET  /api/users/:id/recommendations    │
│       /:movieId/explain                 │
└────────────┬────────────────────────────┘
             │  carga al arrancar 
             ▼
┌─────────────────────────────────────────┐
│         model_cache.pkl (~130 MB)       │
│  trainset       (Surprise trainset)     │
│  best_model     (KNNWithMeans fitted)   │
│  user_neighbors (top-50 vecinos/usuario)│
│  movies_dict    (~27k películas)        │
│  user_ratings   (historial por usuario) │
│  best_params    (k=50, gamma=200, ...)  │
└─────────────────────────────────────────┘
             │  se construye con:
             ▼
┌─────────────────────────────────────────┐
│    python user_user.py build            │
│  (lee rating.csv + movie.csv de dat/)   │
└─────────────────────────────────────────┘

Archivos de persistencia en tiempo de ejecución:
  user_ratings_extra.json  → ratings nuevos y usuarios modificados
```

---

## 3. El dataset

**MovieLens 20M** — 20 millones de ratings de ~138k usuarios sobre ~27k películas.

| Archivo | Filas | Columnas clave |
|---------|-------|----------------|
| `rating.csv` | 20,000,263 | userId, movieId, rating (0.5–5.0), timestamp |
| `movie.csv` | 27,278 | movieId, title, genres (pipe-separated) |

> Solo se usan `rating.csv` y `movie.csv` en el pipeline del modelo.

---

## 4. Pipeline de datos

El pipeline se ejecuta una sola vez con `python user_user.py build`.

### Paso 1 — Muestreo estratificado (igual que Ítem-Ítem)

Se selecciona el **30%** de usuarios (`SAMPLE_SIZE = 0.3`) estratificado por **cantidad de ratings por usuario** usando bins de 100 en 100, garantizando representación de todos los niveles de actividad.

```
Usuarios estratificados por bins de ratings:
  [1, 100]   → 30% de esos usuarios
  [101, 200] → 30% de esos usuarios
  ...
  [max+1, ∞] → 30% de esos usuarios

Semilla: RANDOM_SEED = 505 (reproducible, idéntico a Ítem-Ítem)
```

**Resultado:** ~41,562 usuarios (de 138,493 totales), ~5.9M ratings.

> **¿Por qué el mismo muestreo que Ítem-Ítem?** Para que los usuarios disponibles para login sean los mismos en ambos modelos. Un usuario que existe en Ítem-Ítem también existe en el historial de Usuario-Usuario.

### Paso 2 — Filtro adicional para Surprise

A diferencia de Ítem-Ítem, User-User calcula similitudes entre **usuarios**, no entre películas. Con 41k usuarios la matriz de similitud tendría 41k × 41k = **1.7 billones de valores (~13 GB de RAM)**, lo que es inviable.

Por eso se reduce a los **2000 usuarios más activos**:

```python
MIN_PER_USER  = 20    # mínimo 20 ratings por usuario
MIN_PER_MOVIE = 10    # mínimo 10 ratings por película
N_USERS       = 2000  # top usuarios más activos

# Resultado: 1.8M ratings, 2000 usuarios, 9818 películas
# Densidad: 9.3% (mucho más densa que Ítem-Ítem 1.1%)
```

> **La alta densidad (9.3%) es la clave del buen rendimiento.** User-User necesita usuarios con muchos ratings en común para calcular similitudes confiables. Al filtrar a los 2000 más activos, cada usuario tiene suficiente historial para encontrar vecinos de calidad.

### Paso 3 — Entrenamiento con Surprise (build_full_trainset)

A diferencia de la experimentación (donde se usa train_test_split 80/20 para evaluar MAE/RMSE), en producción se usa **build_full_trainset()** que aprovecha el 100% de los datos:

```python
trainset = dataset.build_full_trainset()
# No hay split — todos los ratings se usan para predecir
```

El modelo KNNWithMeans calcula internamente la matriz de similitud usuario-usuario usando pearson_baseline con shrinkage (McLaughlin).

### Paso 4 — Precálculo de vecinos

Para acelerar el modal "¿Por qué?" se precalculan los top-50 vecinos de cada usuario al construir el caché:

```python
for inner_uid in trainset.all_users():
    sim_row = algo.sim[inner_uid]
    top_50  = np.argsort(sim_row)[::-1][1:51]
    user_neighbors[raw_uid] = [
        {"userId": vecino, "similarity": sim}
        for vecino, sim in top_50 if sim > 0
    ]
```

Esto evita calcular similitudes en tiempo de petición.

### ¿Por qué algunas películas top-1 no muestran vecinos en el modal?

Es posible que una película aparezca como **#1 en recomendaciones** y sin embargo el modal "¿Por qué?" muestre "Ninguno de tus usuarios similares calificó esta película". Esto no es un error — es una consecuencia de cómo funciona el modelo.

**La predicción y la explicación usan fuentes distintas:**

- **Predicción** (): Surprise usa internamente **todos los vecinos posibles del trainset**. Si al menos  vecinos calificaron la película, genera una predicción válida — aunque esos vecinos no estén entre los top-50 precalculados.

- **Explicación** (): Solo consulta , que contiene los **top-50 vecinos precalculados**. Si ninguno de esos 50 calificó la película, el modal queda vacío.

**¿Cómo puede pasar esto?**

Los vecinos que calificaron una película poco vista pueden ser el vecino #51, #72 o #130 — fuera del top-50. Surprise los encuentra en el trainset completo para predecir, pero el modal solo muestra los top-50.

**Ejemplo concreto:**



**¿Por qué ocurre más en películas obscuras?**

Las películas populares (Shawshank, Pulp Fiction) fueron vistas por cientos de usuarios, así que es muy probable que varios de los top-50 vecinos las hayan calificado. Las películas obscuras solo las vieron unos pocos usuarios, y esos usuarios pueden no estar entre los top-50 vecinos más similares.

**¿Cuándo sí aparecen vecinos?**

Cuando la película es suficientemente popular como para que al menos uno de los top-50 vecinos la haya calificado. En la práctica, las películas del top-5 de recomendaciones casi siempre tienen vecinos visibles porque el modelo tiende a recomendar películas bien valoradas por muchos usuarios.

---

## 5. La métrica de similitud

### Pearson Baseline con Significance Weighting (McLaughlin)

Esta es la **única métrica** del modelo User-User (a diferencia de Ítem-Ítem que ofrece tres). Fue elegida como mejor configuración tras los experimentos:

| Métrica | MAE | RMSE |
|---------|-----|------|
| Jaccard | 0.7617 | 0.9882 |
| Coseno | 0.6297 | 0.8126 |
| Pearson | 0.6259 | 0.8163 |
| **Pearson Baseline γ=200** | **0.5880** | **0.7718** |

**¿Qué hace Pearson Baseline?**

Calcula la correlación de Pearson entre dos usuarios, pero primero ajusta los ratings por las medias de usuario Y película (baseline), eliminando sesgos de escala:

```
sim(u, v) = Σ (r_u,i - b_u,i)(r_v,i - b_v,i)  /
            √[Σ(r_u,i - b_u,i)²] √[Σ(r_v,i - b_v,i)²]

donde b_u,i = media_global + desviación_usuario_u + desviación_pelicula_i
```

**¿Qué añade el Significance Weighting (McLaughlin)?**

Penaliza similitudes calculadas sobre pocos ítems en común (ver sección 6).

**En Surprise:** se implementa como `pearson_baseline` con parámetro `shrinkage=gamma`:

```python
KNNWithMeans(
    k=50, min_k=5,
    sim_options={
        "name":      "pearson_baseline",
        "user_based": True,
        "shrinkage":  200   # gamma de McLaughlin
    }
)
```

---

## 6. Pesos de significancia McLaughlin

**Problema:** Dos usuarios pueden tener similitud alta calculada sobre muy pocas películas en común. Por ejemplo, `sim(u, v) = 0.95` pero solo valoraron 2 películas juntos — esa similitud no es estadísticamente confiable.

**Solución McLaughlin:** Penalizar similitudes con pocos co-ratings usando `shrinkage = gamma`:

```
sim_ajustada(u, v) = sim(u, v) × n_uv / (n_uv + γ)
```

Donde `n_uv` es el número de ítems co-valorados entre los usuarios `u` y `v`.

**Ejemplo con γ = 200:**

| n_uv (ítems en común) | factor | sim original | sim ajustada |
|----------------------|--------|--------------|--------------|
| 10 | 10/210 = 0.05 | 0.90 | 0.043 |
| 50 | 50/250 = 0.20 | 0.90 | 0.180 |
| 100 | 100/300 = 0.33 | 0.90 | 0.300 |
| 200 | 200/400 = 0.50 | 0.90 | 0.450 |
| 400 | 400/600 = 0.67 | 0.90 | 0.600 |

Con γ=200 se requieren **más de 200 ítems en común** para que la similitud no sea penalizada significativamente. Este valor alto fue el mejor en los experimentos porque los 2000 usuarios más activos tienen suficientes ratings en común para justificarlo.

**Implementado directamente en Surprise** como parámetro `shrinkage`:

```python
sim_options={"name": "pearson_baseline", "shrinkage": 200}
```

---

## 7. El modelo de predicción

### Fórmula central (KNNWithMeans — User-User)

```
           mean_u + Σ sim(u,v) · (r_v,i - mean_v)
pred(u,i) = ─────────────────────────────────────────
                        Σ |sim(u,v)|
```

Donde:
- `u` = usuario activo
- `i` = película a predecir
- `v` ∈ vecinos = usuarios similares a `u` que valoraron la película `i`
- `sim(u,v)` = similitud Pearson baseline entre `u` y `v`
- `r_v,i` = rating que el vecino `v` dio a la película `i`
- `mean_u` = rating promedio del usuario `u`
- `mean_v` = rating promedio del vecino `v`

**La idea:** Se parte de la media del usuario activo (`mean_u`) y se ajusta según si sus vecinos tendieron a valorar la película `i` por encima o por debajo de sus propias medias.

### Parámetros del mejor modelo (según experimentos)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `k` | 50 | Máximo de vecinos a usar |
| `min_k` | 5 | Mínimo de vecinos con rating para hacer predicción |
| `gamma` | 200 | Factor de shrinkage McLaughlin |
| MAE | 0.5880 | Error absoluto medio en test set |
| RMSE | 0.7718 | Raíz del error cuadrático medio en test set |

### Proceso de predicción en producción

```python
# Surprise maneja todo internamente
pred = algo.predict(userId, movieId)
# pred.est = rating predicho (float entre 0.5 y 5.0)
# pred.details["was_impossible"] = True si no hay vecinos suficientes
```

---

## 8. Algoritmo de recomendaciones

El endpoint de recomendaciones itera sobre **todas las películas** del trainset y predice el rating para cada una:

```python
preds = []
for inner_iid in trainset.all_items():
    raw_iid = trainset.to_raw_iid(inner_iid)
    if raw_iid in rated:          # saltar películas ya vistas
        continue
    pred = algo.predict(user_id, raw_iid)
    if not pred.details["was_impossible"]:
        preds.append((raw_iid, pred.est))

preds.sort(key=lambda x: x[1], reverse=True)
return preds[:limit]
```

**¿Por qué no usa candidatos en 2 fases como Ítem-Ítem?**

Ítem-Ítem necesita las 2 fases porque la matriz de similitud es de 8,798 × 8,798 y el scoring directo sería muy lento. User-User usa Surprise que maneja el KNN eficientemente internamente: para cada película, busca directamente los vecinos que la valoraron usando la matriz de similitud pre-calculada. Con ~9,800 películas y el modelo en memoria, tarda ~3-5 segundos por usuario.

### Fallback para usuarios nuevos

Si el usuario no está en el modelo (usuario recién creado), se devuelven las películas más populares:

```python
if not trainset.knows_user(user_id):
    # Películas con mayor rating promedio global
    pop = ii_media_items.nlargest(limit * 3).index
    return películas_populares
```

---

## 9. Algoritmo de explicación

El endpoint `GET /api/users/:id/recommendations/:movieId/explain` muestra los **vecinos que influyeron** en la predicción:

```python
# Vecinos precalculados del usuario
precomputed = user_neighbors.get(user_id, [])

neighbors_out = []
for nb in precomputed:
    # ¿Calificó esta película el vecino?
    nb_rating = trainset.ur[nb_inner][movie_id]
    if nb_rating is None:
        continue

    # Películas en común (evidencia de similitud)
    shared = [
        película
        for película in (películas_usuario ∩ películas_vecino)[:5]
    ]

    neighbors_out.append({
        "userId":         nb_uid,
        "similarity":     sim_val,        # % similitud con el usuario
        "ratingForMovie": nb_rating,      # cómo calificó ESTA película
        "sharedMovies":   shared,         # películas en común
    })
```

**Lo que ve el usuario en el modal "¿Por qué?":**

| Campo | Descripción |
|-------|-------------|
| `neighborUsers` | Lista de usuarios similares que calificaron la película |
| `similarity` | % de similitud Pearson baseline con ese vecino |
| `ratingForMovie` | Rating que ese vecino le dio a la película recomendada |
| `sharedMovies` | Películas que tú y el vecino calificaron — prueba de similitud |
| `userRatingsEvidence` | Tus últimas 10 películas calificadas (contexto del usuario) |

---

## 10. Estructura del caché

El archivo `model_cache.pkl` contiene:

| Clave | Tipo | Descripción |
|-------|------|-------------|
| `trainset` | `Surprise.Trainset` | Estructura interna de Surprise con todos los ratings |
| `best_model` | `KNNWithMeans` | Modelo entrenado listo para predecir |
| `user_neighbors` | `dict` | `{userId: [{userId, similarity}]}` — top-50 vecinos precalculados |
| `movies_dict` | `dict` | `{movieId: {title, genres}}` — catálogo completo ~27k películas |
| `user_ratings` | `dict` | `{userId: [{movieId, rating, ratedAt}]}` — historial ~41k usuarios |
| `best_params` | `dict` | `{similarity, k, min_k, gamma, mae, rmse}` |

**Tamaño:** ~130 MB en disco, ~500 MB en RAM.

> Mucho más ligero que Ítem-Ítem (~3-5 GB) porque no almacena matrices de similitud densas. La similitud se calcula internamente por Surprise y solo se guarda el resultado (los vecinos precalculados).

---

## 11. Persistencia en tiempo de ejecución

### `user_ratings_extra.json`

Guarda ratings nuevos de usuarios existentes Y todos los ratings de usuarios nuevos.

**Lógica de `_save_extra()`:**

```python
def _save_extra():
    extra = {}
    for uid, rats in user_ratings.items():
        if uid not in _original_users:
            # Usuario nuevo — guardar todo
            extra[str(uid)] = rats
        else:
            # Usuario existente — solo si tiene ratings nuevos
            orig_ids = _original_movie_ids.get(uid, set())
            nuevos   = [r for r in rats if r["movieId"] not in orig_ids]
            if nuevos:
                extra[str(uid)] = rats  # guardar todos sus ratings
```

Al reiniciar el servidor, este archivo se carga y restaura los ratings de la sesión anterior.

---

## 12. Endpoints de la API

Base URL: `http://localhost:8000/api`

---

### `GET /api/health`

Verifica que el servidor está activo y muestra el estado de los modelos.

**Response 200:**
```json
{
  "status":    "ok",
  "users":     41562,
  "user_user": "loaded",
  "item_item": "loaded",
  "uu_params": {
    "similarity": "pearson_baseline",
    "k": 50,
    "min_k": 5,
    "gamma": 200,
    "mae": 0.588,
    "rmse": 0.7718
  }
}
```

---

### `GET /api/users/{id}`

Autentica un usuario por su ID.

**Response 200:**
```json
{
  "user": {
    "userId": 1,
    "displayName": "Usuario 1",
    "totalRatings": 232,
    "joinedAt": "1996-09-19"
  }
}
```

**Response 404:**
```json
{ "detail": "Usuario 1 no encontrado. IDs validos: [2, 5, 8, ...]" }
```

---

### `POST /api/users`

Crea un nuevo usuario con ratings iniciales e integra al modelo User-User via re-fit.

**Request:**
```json
{
  "ratings": [
    { "movieId": 593,  "rating": 5.0 },
    { "movieId": 260,  "rating": 4.0 },
    { "movieId": 356,  "rating": 3.5 }
  ]
}
```

**Response 200:**
```json
{
  "user": {
    "userId": 999999,
    "displayName": "Usuario 999999",
    "totalRatings": 3,
    "joinedAt": "2024-01-01"
  }
}
```

**Proceso interno:**
1. Asigna nuevo ID
2. Guarda ratings en `user_ratings`
3. Re-fit del modelo Surprise incluyendo al nuevo usuario (~2-3 min)
4. Calcula y guarda sus vecinos en `user_neighbors`
5. Persiste en `user_ratings_extra.json`

---

### `GET /api/users/{id}/ratings`

Historial de ratings del usuario.

**Response 200:**
```json
{
  "ratings": [
    {
      "movieId": 593,
      "title": "Silence of the Lambs, The (1991)",
      "genres": ["Crime", "Horror", "Thriller"],
      "rating": 5.0,
      "ratedAt": "1996-09-19"
    }
  ]
}
```

---

### `POST /api/users/{id}/ratings`

Guarda un nuevo rating. Persiste en `user_ratings_extra.json`.

**Request:**
```json
{ "movieId": 296, "rating": 4.5 }
```

**Response 200:**
```json
{
  "rating": {
    "movieId": 296,
    "title": "Pulp Fiction (1994)",
    "genres": ["Comedy", "Crime", "Drama", "Thriller"],
    "rating": 4.5,
    "ratedAt": "2024-01-01"
  }
}
```

---

### `POST /api/users/{id}/refresh`

Re-entrena el modelo User-User incluyendo los ratings actuales del usuario.

**Response 200:**
```json
{ "status": "ok" }
```

**Uso:** Después de calificar varias películas nuevas, el usuario puede presionar "↻ Actualizar" para que las nuevas valoraciones influyan en las recomendaciones (~2-3 segundos).

---

### `GET /api/users/{id}/recommendations`

Genera recomendaciones personalizadas.

**Query parameters:**

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `model` | `user-user` | `user-user` o `item-item` |
| `limit` | `10` | Número de recomendaciones |

**Ejemplo:** `GET /api/users/1/recommendations?model=user-user&limit=10`

**Response 200:**
```json
{
  "recommendations": [
    {
      "rank": 1,
      "movieId": 318,
      "title": "Shawshank Redemption, The (1994)",
      "genres": ["Crime", "Drama"],
      "predictedRating": 4.73
    }
  ]
}
```

---

### `GET /api/users/{id}/recommendations/{movieId}/explain`

Explica por qué se recomienda una película.

**Ejemplo:** `GET /api/users/1/recommendations/318/explain?model=user-user`

**Response 200:**
```json
{
  "explanation": {
    "userId": 1,
    "movieId": 318,
    "movieTitle": "Shawshank Redemption, The (1994)",
    "movieGenres": ["Crime", "Drama"],
    "movieAvgRating": 4.18,
    "predictedRating": 4.73,
    "modelUsed": "User-User — pearson_baseline, k=50, gamma=200",
    "neighborUsers": [
      {
        "userId": 452,
        "displayName": "Usuario 452",
        "similarity": 0.8923,
        "ratingForMovie": 5.0,
        "sharedMovies": [
          {
            "movieId": 296,
            "title": "Pulp Fiction (1994)",
            "userRating": 5.0,
            "neighborRating": 5.0
          }
        ]
      }
    ],
    "neighborItems": [],
    "userRatingsEvidence": [
      {
        "movieId": 296,
        "title": "Pulp Fiction (1994)",
        "rating": 5.0,
        "similarity": 0.5
      }
    ]
  }
}
```

---

### `GET /api/movies/search?q={query}`

Búsqueda de películas por título (máximo 20 resultados).

**Response 200:**
```json
{
  "movies": [
    {
      "movieId": 79132,
      "title": "Inception (2010)",
      "genres": ["Action", "Crime", "Drama", "Mystery", "Sci-Fi", "Thriller"]
    }
  ]
}
```

---

## 13. Flujo completo end-to-end

```
1. USUARIO ABRE http://localhost:3000
   └─► LoginPage renderiza

2. USUARIO INGRESA SU ID Y PULSA "Entrar"
   └─► loginUser(id)
   └─► GET http://localhost:8000/api/users/:id
   └─► Backend busca en user_ratings (pkl) + user_ratings_extra.json
   └─► Responde { user: { userId, totalRatings, joinedAt } }
   └─► Frontend navega a /dashboard

3. DASHBOARD CARGA
   └─► useEffect → loadData()
       ├─► getUserRatings(userId)
       │   └─► GET /api/users/:id/ratings
       │   └─► Backend retorna historial con títulos y géneros
       │   └─► Se muestran en columna izquierda
       │
       └─► getRecommendations(userId, { model: "user-user", limit: 10 })
           └─► GET /api/users/:id/recommendations?model=user-user&limit=10
           └─► Backend:
               ├── trainset.knows_user(userId) → ¿está en el modelo?
               ├── Para cada película no vista: algo.predict(userId, movieId)
               ├── Ordena por rating predicho descendente
               └── Retorna top 10
           └─► Se muestran en columna derecha con rank y predictedRating

4. USUARIO PULSA "¿Por qué?" EN UNA RECOMENDACIÓN
   └─► handleExplain(movie)
   └─► explainRecommendation(userId, movieId, { model: "user-user" })
   └─► GET /api/users/:id/recommendations/:movieId/explain?model=user-user
   └─► Backend:
       ├── Busca vecinos precalculados en user_neighbors[userId]
       ├── Para cada vecino: ¿calificó esta película?
       ├── Encuentra películas en común como evidencia
       └── Retorna neighborUsers con similitud + sharedMovies
   └─► ExplanationModal abre con tabs:
       ├── "Usuarios vecinos" → usuarios similares que calificaron la película
       └── "Tus valoraciones" → tus últimas películas calificadas

5. USUARIO VALORA UNA RECOMENDACIÓN
   └─► handleRateConfirm(movieId, rating)
   └─► rateMovie(userId, movieId, rating)
   └─► POST /api/users/:id/ratings  { movieId, rating }
   └─► Backend guarda en memory + user_ratings_extra.json (persiste)
   └─► La película desaparece de recomendaciones
   └─► Aparece en el historial de ratings

6. USUARIO PULSA "↻ Actualizar"
   └─► refreshRecommendations(userId)
   └─► POST /api/users/:id/refresh
   └─► Backend re-entrena modelo con ratings actuales (~2-3 seg)
   └─► getRecommendations() con nuevas predicciones
   └─► Lista actualizada con los nuevos ratings incluidos

7. USUARIO CAMBIA A ITEM-ITEM
   └─► Toggle en el panel de modelo activo
   └─► onModelParamsChange({ modelType: "item-item" })
   └─► loadData({ modelType: "item-item" })
   └─► GET /api/users/:id/recommendations?model=item-item&limit=10
   └─► Backend usa el modelo ítem-ítem para generar nuevas recomendaciones
```

---

## 14. Cómo correr el sistema

### Requisitos

```bash
pip install fastapi uvicorn scikit-surprise pandas numpy scikit-learn
```

### Paso 1 — Construir el caché User-User (solo la primera vez)

```bash
cd Movie_Recomendation/backend
python user_user.py build
```

Proceso (~45 minutos):
```
========================================================
USER-USER — Generando model_cache.pkl
========================================================
Cargando datos...
  rating: 20,000,263 | movie: 27,278
Muestreo estratificado 30%...
  41,562 usuarios | 5,996,602 ratings
Filtrando top 2000 usuarios mas activos...
  1,817,207 ratings | 2000 usuarios | 9818 peliculas | densidad 9.3%
Construyendo catalogo de peliculas (~27k)...
  27,278 peliculas cargadas
Entrenando pearson_baseline k=50 gamma=200...
  [Surprise calcula matriz de similitud 2000×2000...]
Precalculando vecinos para 2000 usuarios...

========================================================
model_cache.pkl listo
  Usuarios en modelo   : 2,000
  Usuarios en historial: 41,562
  Peliculas en catalogo: 27,278
  Modelo: pearson_baseline k=50 gamma=200
========================================================
```

### Paso 2 — Iniciar el servidor unificado

```bash
cd Movie_Recomendation/backend
python -m uvicorn main:app --reload --port 8000
```

Al arrancar (~30 segundos):
```
Cargando model_cache.pkl (User-User)...
  User-User OK — 41,562 usuarios
Cargando item_item_cache.pkl (Item-Item)...
  Item-Item OK — 41,562 usuarios | matriz (41562, 8798)
Servidor listo. Usuarios: 41,562
```

### Paso 3 — Frontend

```bash
cd Movie_Recomendation/frontend
npm start
```

### Probar la API directamente

```bash
# Health
curl http://localhost:8000/api/health

# Login (usar un ID de los mostrados en el error si el usuario no existe)
curl http://localhost:8000/api/users/1

# Recomendaciones User-User
curl "http://localhost:8000/api/users/1/recommendations?model=user-user&limit=5"

# Explicación
curl "http://localhost:8000/api/users/1/recommendations/318/explain?model=user-user"
```

### Documentación interactiva (Swagger)

```
http://localhost:8000/docs
```

---

## 15. Diferencias con el modelo Ítem-Ítem

| Aspecto | User-User | Ítem-Ítem |
|---------|-----------|-----------|
| **Similitud entre** | Usuarios | Películas |
| **Matriz de similitud** | 2000×2000 (~16 MB) | 8798×8798 (~310 MB × 3) |
| **Tamaño del caché** | ~130 MB | ~2-3 GB |
| **Usuarios en modelo** | 2000 (los más activos) | 41,562 (todos) |
| **Usuarios en historial** | 41,562 | 41,562 |
| **Tiempo de build** | ~45 minutos | ~15-30 minutos |
| **Tiempo de carga** | ~5 segundos | ~60 segundos |
| **Tiempo por recomendación** | ~3-5 segundos | ~0.1 segundos |
| **MAE mejor config** | 0.588 (Pearson+McLaughlin) | ~0.58 (Pearson) |
| **Métricas disponibles** | Solo Pearson Baseline | Coseno, Pearson, Jaccard |
| **Explicación muestra** | Usuarios vecinos similares | Películas similares del historial |
| **Usuarios nuevos** | Re-fit (~2 min) o fallback | Funciona inmediatamente |

---

## 16. Limitaciones y consideraciones

### Usuarios cubiertos
- Solo los **2000 usuarios más activos** tienen recomendaciones User-User reales.
- Los otros ~39,562 usuarios del historial pueden hacer login y ver su historial, pero sus recomendaciones User-User usan el fallback de popularidad.
- **Solución:** usar el toggle Item-Item para esos usuarios — Item-Item funciona para todos.

### Nuevos ratings de usuarios en el modelo
- Los ratings añadidos en tiempo de ejecución se guardan en memoria y disco.
- El modelo Surprise NO se actualiza automáticamente — es necesario presionar "↻ Actualizar" para que influyan en las predicciones.
- El botón re-entrena el modelo (~2-3 segundos) incluyendo los nuevos ratings.

### Nuevos usuarios
- Se integran al modelo via re-fit al crearlos.
- El re-fit tarda ~2-3 minutos porque re-calcula la matriz de similitud completa.
- Con solo 3-5 ratings iniciales, la calidad de las recomendaciones es menor que para usuarios con cientos de ratings.

### Escalabilidad
- El sistema está diseñado para desarrollo y demostración.
- Para producción: usar ALS (Alternating Least Squares), embeddings de usuarios, o aproximaciones con FAISS.

### Reproducibilidad
- `RANDOM_SEED = 505` garantiza muestreo idéntico en cada build.
- Los resultados son deterministas dado el mismo dataset y seed.
