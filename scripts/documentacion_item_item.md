# Sistema de Recomendación Ítem-Ítem — Documentación Completa

> **Archivo:** `backend/Item-Item.py`
> **Dataset:** MovieLens 20M
> **Stack:** Python · NumPy · Pandas · scikit-learn · FastAPI · React

---

## Tabla de Contenidos

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Arquitectura general](#2-arquitectura-general)
3. [El dataset](#3-el-dataset)
4. [Pipeline de datos](#4-pipeline-de-datos)
5. [Las 3 métricas de similitud](#5-las-3-métricas-de-similitud)
6. [Pesos de significancia McLaughlin](#6-pesos-de-significancia-mclaughlin)
7. [El modelo de predicción](#7-el-modelo-de-predicción)
8. [Algoritmo de recomendaciones (2 fases)](#8-algoritmo-de-recomendaciones-2-fases)
9. [Algoritmo de explicación](#9-algoritmo-de-explicación)
10. [Estructura del caché](#10-estructura-del-caché)
11. [Persistencia en tiempo de ejecución](#11-persistencia-en-tiempo-de-ejecución)
12. [Los 7 endpoints de la API](#12-los-7-endpoints-de-la-api)
13. [Parámetros configurables del frontend](#13-parámetros-configurables-del-frontend)
14. [Flujo completo end-to-end](#14-flujo-completo-end-to-end)
15. [Cómo correr el sistema](#15-cómo-correr-el-sistema)
16. [Diferencias con el modelo Usuario-Usuario](#16-diferencias-con-el-modelo-usuario-usuario)
17. [Limitaciones y consideraciones](#17-limitaciones-y-consideraciones)

---

## 1. Resumen ejecutivo

El sistema implementa **filtrado colaborativo ítem-ítem** sobre el dataset MovieLens 20M. Dado un usuario, predice el rating que daría a películas que no ha visto comparando dichas películas con las que sí ha valorado, usando la similitud entre ítems calculada sobre el comportamiento histórico de toda la comunidad de usuarios.

El resultado es un servidor FastAPI (`Item-Item.py`) que expone 7 endpoints REST consumidos por una aplicación React. El modelo soporta tres métricas de similitud (coseno, Pearson, Jaccard), dos modos de selección de vecinos (top-k y umbral) y ponderación de significancia McLaughlin.

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
│      BACKEND FastAPI (:8000)            │
│  POST /api/login                        │
│  POST /api/users                        │
│  GET  /api/users/:id/ratings            │
│  POST /api/users/:id/ratings            │
│  GET  /api/movies/search                │
│  GET  /api/users/:id/recommendations    │
│  GET  /api/users/:id/recommendations    │
│       /:movieId/explain                 │
└────────────┬────────────────────────────┘
             │  carga al arrancar (~60s)
             ▼
┌─────────────────────────────────────────┐
│         item_item_cache.pkl (~2-3 GB)   │
│  matriz_usuario_item  (41k × 8.8k)     │
│  sim_coseno / sim_pearson / sim_jaccard │
│  matriz_co_ratings                      │
│  media_por_item │ rating_promedio_global│
│  pesos_25 (McLaughlin γ=25)            │
│  movies_dict │ user_ratings             │
└─────────────────────────────────────────┘
             │  se construye con:
             ▼
┌─────────────────────────────────────────┐
│    python Item-Item.py build            │
│  (lee rating.csv + movie.csv de dat/)   │
└─────────────────────────────────────────┘

Archivos de persistencia en tiempo de ejecución:
  user_ratings_extra.json   → ratings guardados en la sesión
  nuevos_usuarios.json      → usuarios creados vía API
```

---

## 3. El dataset

**MovieLens 20M** — 20 millones de ratings de ~138k usuarios sobre ~27k películas.

| Archivo | Filas | Columnas clave |
|---------|-------|----------------|
| `rating.csv` | 20,000,263 | userId, movieId, rating (0.5–5.0), timestamp |
| `movie.csv` | 27,278 | movieId, title, genres (pipe-separated) |
| `tag.csv` | 465,564 | userId, movieId, tag, timestamp |
| `link.csv` | 27,278 | movieId, imdbId, tmdbId |
| `genome_scores.csv` | 11,709,768 | movieId, tagId, relevance |
| `genome_tags.csv` | 1,128 | tagId, tag |

> Solo se usan `rating.csv` y `movie.csv` en el pipeline del modelo.

---

## 4. Pipeline de datos

El pipeline se ejecuta una sola vez con `python Item-Item.py build` y el resultado queda en el caché.

### Paso 1 — Muestreo estratificado

Se selecciona el **30%** de usuarios (`SAMPLE_SIZE = 0.3`) pero no aleatoriamente: se estratifica por **cantidad de ratings por usuario** usando bins de 100 en 100, garantizando que todos los niveles de actividad estén representados.

```
Usuarios estratificados por bins de ratings:
  [1, 100]   → 30% de esos usuarios
  [101, 200] → 30% de esos usuarios
  [201, 300] → 30% de esos usuarios
  ...
  [max+1, ∞] → 30% de esos usuarios

Semilla: RANDOM_SEED = 505 (reproducible)
```

**Resultado:** ~41,562 usuarios (de 138,493 totales), ~5.9M ratings.

### Paso 2 — Split Train / Test

```python
train_set, test_set = train_test_split(
    rating_sample, test_size=0.3, random_state=505
)
# Train: ~4,197,621 ratings
# Test:  ~1,798,981 ratings
```

### Paso 3 — Filtrado de películas frecuentes

Se eliminan las películas con menos de 20 ratings en el train set para evitar similitudes ruidosas:

```python
MIN_RATINGS_PELICULA = 20
peliculas_frecuentes = train_set.groupby('movieId').size()[lambda s: s >= 20].index
# Resultado: 8,798 películas con >= 20 ratings
```

### Paso 4 — Matriz usuario-ítem

```python
matriz_usuario_item = entrenamiento_filtrado.pivot_table(
    index='userId', columns='movieId', values='rating'
)
# Forma: 41,562 usuarios × 8,798 películas
# Densidad: ~1.13% (mayoría de celdas son NaN)
```

Esta es la estructura central del modelo. Cada fila es un usuario, cada columna una película, cada celda es el rating que dio ese usuario (o NaN si no la vio).

### Variables globales derivadas del train set

```python
media_por_item         = train_set.groupby('movieId')['rating'].mean()
rating_promedio_global = train_set['rating'].mean()  # ≈ 3.527
```

---

## 5. Las 3 métricas de similitud

Todas producen una matriz cuadrada de forma **(8,798 × 8,798)** donde `sim[i][j]` es la similitud entre la película `i` y la película `j`.

### 5.1 Similitud Coseno

Mide el ángulo entre los vectores de ratings de dos películas (rellenando NaN con 0):

```
cos(i, j) = (Σ r_u,i · r_u,j) / (‖r_i‖ · ‖r_j‖)
```

```python
def calcular_similitud_coseno(matriz_ui):
    M = matriz_ui.fillna(0).values.T.astype(np.float32)  # shape: (8798, 41562)
    return cosine_similarity(M)  # shape: (8798, 8798)
```

**Características:**
- Rápida de calcular
- No ajusta por el nivel de ratings (usuarios "optimistas" vs "pesimistas")
- Buena para matrices muy dispersas

### 5.2 Similitud Pearson

Primero **centra cada ítem por su media** (elimina el sesgo de escala) y luego calcula coseno sobre la matriz centrada:

```
pearson(i, j) = cos(i - mean_i, j - mean_j)
```

```python
def calcular_similitud_pearson(matriz_ui):
    media_items     = matriz_ui.mean(axis=0)
    matriz_centrada = matriz_ui.subtract(media_items, axis=1)
    M = matriz_centrada.fillna(0).values.T.astype(np.float32)
    return cosine_similarity(M)
```

**Características:**
- Compensa el sesgo de escala de usuarios (un usuario que siempre da 4-5 estrellas vs uno que da 1-3)
- **Mejor métrica en general** (RMSE ~0.76 vs ~0.80 del coseno)
- **Valor por defecto del sistema**

### 5.3 Similitud Jaccard

Mide la **proporción de usuarios en común** que valoraron ambas películas, ignorando el valor del rating:

```
jaccard(i, j) = |{usuarios que valoraron i Y j}| / |{usuarios que valoraron i O j}|
```

```python
def calcular_similitud_jaccard(matriz_ui):
    B = matriz_ui.notna().values.astype(np.float32)  # matriz binaria
    interseccion = B.T @ B                            # co-ratings
    union = conteo_i + conteo_j - interseccion
    similitud = interseccion / union
    return sim_jaccard, matriz_co_ratings
```

**Características:**
- No usa los valores de rating, solo presencia/ausencia
- Menos precisa para predicción (RMSE ~0.84)
- **Produce también `matriz_co_ratings`**, necesaria para los pesos McLaughlin

### Comparación de rendimiento (muestra 2000, top_k=20)

| Métrica | RMSE | MAE |
|---------|------|-----|
| Coseno | ~0.80 | ~0.62 |
| **Pearson** | **~0.76** | **~0.58** |
| Jaccard | ~0.84 | ~0.65 |

---

## 6. Pesos de significancia McLaughlin

**Problema:** Dos películas pueden tener una similitud alta pero calculada sobre muy pocos usuarios en común (ruido estadístico). Por ejemplo, `sim(A, B) = 0.95` pero solo 2 usuarios vieron ambas.

**Solución McLaughlin:** Penalizar similitudes calculadas con pocos co-ratings usando un umbral `gamma`:

```
peso(i, j) = min(co_ratings(i, j), γ) / γ
```

```python
def calcular_pesos_mclaughlin(matriz_co_ratings, gamma):
    n = matriz_co_ratings.values.astype(float)
    pesos = np.minimum(n, gamma) / gamma
    return pd.DataFrame(pesos, ...)
```

**Ejemplo con γ = 25:**

| co_ratings(i,j) | peso |
|-----------------|------|
| 0 | 0.00 |
| 5 | 0.20 |
| 10 | 0.40 |
| 25 | 1.00 |
| 50 | 1.00 |

Los pesos se **multiplican** por la similitud antes de calcular la predicción:

```python
similitudes_ajustadas[vecino] = sim[item, vecino] * peso[item, vecino]
```

**En el sistema:**
- `pesos_25` está precalculado en el caché (gamma=25)
- Si el usuario elige un gamma diferente vía `significanceAlpha`, se recalcula al vuelo
- Se activa con `significanceWeighting=true` en el frontend

---

## 7. El modelo de predicción

### Fórmula central (SR ítem-ítem ajustado por media)

```
        Σ sim(i, n) · (r_u,n - mean_n)
pred(u, i) = mean_i + ──────────────────────────
                       Σ |sim(i, n)|
```

Donde:
- `u` = usuario
- `i` = película a predecir
- `n` ∈ vecinos = películas que `u` ya valoró y son similares a `i`
- `sim(i, n)` = similitud entre la película objetivo y el vecino
- `r_u,n` = rating que el usuario `u` dio a la película `n`
- `mean_n` = rating promedio global de la película `n`
- `mean_i` = rating promedio global de la película `i`

**La idea:** La predicción parte de la media de la película objetivo (`mean_i`) y se ajusta según si el usuario tendió a puntuar los ítems similares por encima o por debajo de sus medias.

### Implementación paso a paso (`predecir_rating_con_ratings`)

```python
def predecir_rating_con_ratings(id_item, ratings_usuario_series, sim_matrix, ...):

    # 1. Verificar que el ítem está en la matriz de similitud
    if id_item not in sim_matrix.index:
        return rating_promedio_global

    # 2. Ratings del usuario (excluir el ítem objetivo)
    ratings_usuario = ratings_usuario_series.dropna().drop(id_item, errors='ignore')
    if ratings_usuario.empty:
        return rating_promedio_global

    # 3. Encontrar ítems en común (vistos por el usuario Y en la matriz)
    items_comunes = ratings_usuario.index.intersection(sim_matrix.columns)

    # 4. Obtener similitudes con los ítems comunes
    similitudes = sim_matrix.loc[id_item, items_comunes].copy()

    # 5. Aplicar pesos McLaughlin (opcional)
    if pesos_significancia is not None:
        similitudes *= pesos_significancia.loc[id_item, items_comunes]

    # 6. Seleccionar vecinos según el modo
    if modo_vecinos == 'top_k':
        vecinos = similitudes[similitudes > 0].nlargest(vecinos_k)
    elif modo_vecinos == 'umbral':
        vecinos = similitudes[similitudes >= umbral_similitud]

    # 7. Calcular predicción
    media_i = media_por_item.get(id_item, rating_promedio_global)
    numerador, denominador = 0.0, 0.0
    for id_vecino, sim in vecinos.items():
        media_n         = media_por_item.get(id_vecino, rating_promedio_global)
        rating_centrado = ratings_usuario[id_vecino] - media_n
        numerador   += sim * rating_centrado
        denominador += abs(sim)

    prediccion = media_i + (numerador / denominador)
    return float(np.clip(prediccion, 0.5, 5.0))
```

### Las 3 variantes de la función de predicción

| Función | Diferencia | Cuándo se usa |
|---------|-----------|---------------|
| `predecir_rating()` | Lee directamente `matriz_ui.loc[id_usuario]` | Evaluación del modelo |
| `predecir_rating_con_ratings()` | Recibe `pd.Series` pre-construida | Endpoint de recomendaciones |
| `predecir_rating_explicado()` | Igual + retorna la lista de vecinos usados | Endpoint de explicación |

---

## 8. Algoritmo de recomendaciones (2 fases)

**Problema:** Calcular `predecir_rating` para las 8,798 películas por cada petición sería demasiado lento.

**Solución:** Dos fases que reducen el trabajo a ~100 predicciones:

### Fase 1 — Generación de candidatos (vectorizada, ~ms)

```python
submatrix = sim_matrix.loc[rated_en_sim]       # (n_vistas, 8798)
max_sim   = submatrix.max(axis=0)              # por cada película: max similitud
candidatos = max_sim.drop(labels=all_rated_ids) # excluir ya vistas
top_100    = candidatos.nlargest(100).index    # top 100 candidatos
```

**Lógica:** Las 100 películas con mayor similitud máxima con alguna película ya vista por el usuario.

### Fase 2 — Scoring de candidatos (~100 predicciones)

```python
scores = []
for mid in top_100:
    pred = predecir_rating_con_ratings(mid, ratings_row, sim_matrix, ...)
    scores.append((mid, pred))
scores.sort(key=lambda x: -x[1])
return scores[:limit]
```

### Fallback para usuarios sin historial

```python
populares = cache['media_por_item'].nlargest(limit * 5).index
# Retornar las películas con mayor rating promedio global
```

---

## 9. Algoritmo de explicación

```python
resultado = predecir_rating_explicado(
    id_item=movie_id,
    ratings_usuario_series=ratings_row,
    sim_matrix=sim_matrix,
    ...
)
# resultado = {
#   'prediccion': 4.23,
#   'vecinos': [(id_vecino, sim, rating_usuario, media_vecino), ...]
# }
```

Cada vecino es una película que el usuario ya valoró y que influyó en la predicción:

| Campo del response | Descripción | Tab en frontend |
|-------------------|-------------|-----------------|
| `userRatingsEvidence` | Películas del historial del usuario con su rating y similitud | "Tus valoraciones" |
| `neighborItems` | Las mismas películas con su rating promedio global | "Items vecinos" |
| `neighborUsers` | Siempre `[]` en ítem-ítem | "Usuarios vecinos" (sin datos) |

El campo `modelUsed` describe la configuración exacta usada:
```
"Item-Item — Pearson, k=20"
"Item-Item — Coseno, umbral=0.3, McLaughlin γ=25"
```

---

## 10. Estructura del caché

El archivo `item_item_cache.pkl` contiene un diccionario Python con las siguientes claves:

| Clave | Tipo | Forma | Tamaño aprox. | Descripción |
|-------|------|-------|----------------|-------------|
| `matriz_usuario_item` | `pd.DataFrame` | 41,562 × 8,798 | ~1.4 GB | Ratings del train set (NaN = no vio) |
| `sim_coseno` | `pd.DataFrame` | 8,798 × 8,798 | ~310 MB | Similitud coseno entre películas |
| `sim_pearson` | `pd.DataFrame` | 8,798 × 8,798 | ~310 MB | Similitud Pearson entre películas |
| `sim_jaccard` | `pd.DataFrame` | 8,798 × 8,798 | ~310 MB | Similitud Jaccard entre películas |
| `matriz_co_ratings` | `pd.DataFrame` | 8,798 × 8,798 | ~310 MB | Nº de usuarios que valoraron ambas películas |
| `media_por_item` | `pd.Series` | 8,798 | < 1 MB | Rating promedio por película |
| `rating_promedio_global` | `float` | — | — | Rating promedio global ≈ 3.527 |
| `pesos_25` | `pd.DataFrame` | 8,798 × 8,798 | ~310 MB | Pesos McLaughlin precalculados con γ=25 |
| `movies_dict` | `dict` | ~27,278 entradas | ~10 MB | `{movieId: {title, genres}}` — todos los ~27k |
| `user_ratings` | `dict` | ~41,562 listas | ~200 MB | `{userId: [{movieId, rating, ratedAt}]}` |
| `user_join_dates` | `dict` | ~41,562 entradas | ~5 MB | `{userId: 'YYYY-MM-DD'}` |
| `train_set` | `pd.DataFrame` | ~4.2M filas | ~200 MB | Ratings de entrenamiento |
| `test_set` | `pd.DataFrame` | ~1.8M filas | ~85 MB | Ratings de prueba |

**Total en disco:** ~2–3 GB
**Total en RAM al cargar:** ~3–5 GB

---

## 11. Persistencia en tiempo de ejecución

### `user_ratings_extra.json`

```json
{
  "500001": [
    { "movieId": 593,  "rating": 4.5, "ratedAt": "2026-03-15" }
  ],
  "1": [
    { "movieId": 296, "rating": 5.0, "ratedAt": "2026-03-15" }
  ]
}
```

Al servir recomendaciones, los extras se añaden a la fila del usuario en la matriz antes de correr el algoritmo. Los extras tienen **mayor prioridad** que los del dataset en caso de duplicado.

### `nuevos_usuarios.json`

```json
[
  {
    "userId": 500001,
    "displayName": "Nuevo Usuario",
    "totalRatings": 3,
    "joinedAt": "2026-03-15"
  }
]
```

IDs de nuevos usuarios empiezan en `500001 + posición` para no colisionar con el dataset.

**Escritura segura:** Ambos archivos se escriben bajo `asyncio.Lock`.

---

## 12. Los 7 endpoints de la API

Base URL: `http://localhost:8000/api`

---

### `POST /api/login`

**Request:** `{ "userId": 1 }`

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

---

### `POST /api/users`

Crea un nuevo usuario con ratings iniciales.

**Request:**
```json
{ "ratings": [ { "movieId": 593, "rating": 5.0 }, ... ] }
```

**Response 200:**
```json
{ "user": { "userId": 500001, "displayName": "Nuevo Usuario", "totalRatings": 3, "joinedAt": "2026-03-15" } }
```

---

### `GET /api/users/{id}/ratings`

**Response 200:**
```json
{
  "ratings": [
    { "movieId": 593, "title": "Silence of the Lambs, The (1991)", "genres": ["Crime","Horror","Thriller"], "rating": 5.0, "ratedAt": "1996-09-19" }
  ]
}
```

---

### `POST /api/users/{id}/ratings`

**Request:** `{ "movieId": 296, "rating": 4.5 }`

**Response 200:**
```json
{ "rating": { "movieId": 296, "title": "Pulp Fiction (1994)", "genres": ["Comedy","Crime","Drama","Thriller"], "rating": 4.5, "ratedAt": "2026-03-15" } }
```

Validación: rating debe estar entre 0.5 y 5.0.

---

### `GET /api/movies/search?q={query}`

Búsqueda por substring case-insensitive sobre los ~27k títulos. Máximo 20 resultados.

**Response 200:**
```json
{ "movies": [ { "movieId": 79132, "title": "Inception (2010)", "genres": ["Action","Sci-Fi","Thriller"] } ] }
```

---

### `GET /api/users/{id}/recommendations`

**Query parameters:**

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `similarity` | `pearson` | `pearson`, `cosine`, `jaccard` |
| `neighborMode` | `k` | `k` (top-k) o `threshold` (umbral) |
| `k` | `20` | Número de vecinos |
| `threshold` | `0.3` | Similitud mínima (modo umbral) |
| `significanceWeighting` | `false` | Activar pesos McLaughlin |
| `significanceAlpha` | `50` | Gamma para McLaughlin |
| `limit` | `10` | Recomendaciones a devolver |

**Response 200:**
```json
{
  "recommendations": [
    { "rank": 1, "movieId": 318, "title": "Shawshank Redemption, The (1994)", "genres": ["Crime","Drama"], "predictedRating": 4.7823 }
  ]
}
```

---

### `GET /api/users/{id}/recommendations/{movieId}/explain`

Mismos query params que recomendaciones (sin `limit`).

**Response 200:**
```json
{
  "explanation": {
    "userId": 1,
    "movieId": 318,
    "movieTitle": "Shawshank Redemption, The (1994)",
    "movieGenres": ["Crime", "Drama"],
    "movieAvgRating": 4.1823,
    "predictedRating": 4.7823,
    "modelUsed": "Item-Item — Pearson, k=20",
    "similarityMetric": "pearson",
    "neighborsUsed": 15,
    "userRatingsEvidence": [
      { "movieId": 296, "title": "Pulp Fiction (1994)", "genres": ["Crime","Drama"], "rating": 5.0, "similarity": 0.8234 }
    ],
    "neighborUsers": [],
    "neighborItems": [
      { "movieId": 296, "title": "Pulp Fiction (1994)", "avgRating": 4.1504, "similarity": 0.8234 }
    ]
  }
}
```

---

## 13. Parámetros configurables del frontend

| Control en UI | Nombre interno | Valores | Default |
|---|---|---|---|
| Tipo de modelo | `modelType` | `user-user`, `item-item` | `user-user` |
| Métrica de similitud | `similarity` | `pearson`, `cosine`, `jaccard` | `pearson` |
| Modo de vecinos | `neighborMode` | `k`, `threshold` | `k` |
| Número de vecinos | `k` | 1–500 | `20` |
| Umbral | `threshold` | 0.0–1.0 | `0.3` |
| Ponderación McLaughlin | `significanceWeighting` | `true`, `false` | `false` |
| Gamma McLaughlin | `significanceAlpha` | entero > 0 | `50` |

---

## 14. Flujo completo end-to-end

```
1. USUARIO ABRE http://localhost:3000
   └─► LoginPage renderiza

2. USUARIO INGRESA SU ID Y PULSA "Entrar"
   └─► POST /api/login { userId: 1 }
   └─► Backend busca en cache['user_ratings'] y nuevos_usuarios.json
   └─► Frontend navega a /dashboard

3. DASHBOARD CARGA
   ├─► GET /api/users/1/ratings → historial en columna izquierda
   └─► GET /api/users/1/recommendations?similarity=pearson&k=20&...
       └─► Fase 1: top 100 candidatos vectorizados
       └─► Fase 2: predecir_rating_con_ratings × 100
       └─► Top 10 en columna derecha

4. USUARIO PULSA "¿Por qué?"
   └─► GET /api/users/1/recommendations/318/explain?...
   └─► predecir_rating_explicado() → predicción + vecinos
   └─► ExplanationModal: "Items vecinos" y "Tus valoraciones"

5. USUARIO VALORA UNA RECOMENDACIÓN
   └─► POST /api/users/1/ratings { movieId:318, rating:4.5 }
   └─► Guarda en user_ratings_extra.json
   └─► La película pasa del panel de recomendaciones al historial

6. USUARIO CAMBIA PARÁMETROS Y PULSA "Actualizar"
   └─► Nuevas recomendaciones con la configuración elegida
```

---

## 15. Cómo correr el sistema

Ver [readme_execution.md](../readme_execution.md) en la raíz del proyecto para instrucciones completas de ejecución de ambos modelos y el frontend.

---

## 16. Diferencias con el modelo Usuario-Usuario

| Aspecto | Ítem-Ítem | Usuario-Usuario |
|---------|-----------|-----------------|
| **Similitud entre** | Películas | Usuarios |
| **Matriz de similitud** | 8798×8798 (~310 MB × 3) | 2000×2000 (~16 MB) |
| **Tamaño del caché** | ~2-3 GB | ~130 MB |
| **Usuarios en modelo** | 41,562 (todos) | 2000 (los más activos) |
| **Tiempo de build** | ~15-30 minutos | ~45 minutos |
| **Tiempo de carga** | ~60 segundos | ~5 segundos |
| **Tiempo por recomendación** | ~0.1 segundos | ~3-5 segundos |
| **MAE mejor config** | ~0.58 (Pearson) | 0.588 (Pearson+McLaughlin) |
| **Métricas disponibles** | Coseno, Pearson, Jaccard | Solo Pearson Baseline |
| **Explicación muestra** | Películas similares del historial | Usuarios vecinos similares |
| **Usuarios nuevos** | Funciona inmediatamente | Re-fit (~2 min) |

---

## 17. Limitaciones y consideraciones

### Memoria
- El caché ocupa **~3-5 GB en RAM**. Se necesita un mínimo de 8 GB disponibles.

### Usuarios nuevos
- Sus ratings están en `user_ratings_extra.json`, NO en `matriz_usuario_item`.
- Con < 5 ratings, los resultados son menos precisos.
- Si valora películas con < 20 ratings en el dataset, esas valoraciones no contribuyen.

### Nuevos ratings de usuarios existentes
- Se suman al vector del usuario al predecir, pero no modifican las matrices de similitud.
- Las matrices solo se reconstruyen con `python Item-Item.py build`.

### El parámetro `neighborMode: 'threshold'`
- Un umbral muy bajo (< 0.01) incluye vecinos ruidosos.
- Un umbral muy alto (> 0.5) puede dejar sin vecinos a muchos ítems → devuelve `rating_promedio_global`.
- **Recomendado:** usar `top_k` con `k=20` como punto de partida.

### Reproducibilidad
- `RANDOM_SEED = 505` garantiza muestreo y split idénticos en cada build.
