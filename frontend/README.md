# MovieLens Recommender — Frontend

Cascarón visual para el **Taller 1: Modelos Colaborativos** (MINE4201).
Funciona sin backend usando datos de MovieLens 20M (películas reales) y mocks mínimos.

---

## Inicio rápido

```bash
cd frontend
npm install
npm start
# → http://localhost:3000
```

IDs de demo: **1, 2, 3, 4** (usuarios reales de MovieLens 20M).

---

## Estructura del proyecto

```
frontend/src/
  api/
    api.js              — ÚNICO punto de integración con el backend
  data/
    movies.json         — catálogo real de MovieLens 20M (~110 películas representativas)
  mock/
    mockData.js         — stubs temporales con IDs reales de MovieLens
  components/
    ExplanationModal.jsx
    RatingControl.jsx
  pages/
    LoginPage.jsx
    CreateUserPage.jsx
    DashboardPage.jsx
  App.jsx               — routing + estado del modelo
  index.js
  index.css             — sistema de diseño global
```

---

## Vistas

| Ruta | Página | Descripción |
|---|---|---|
| `/` | LoginPage | Login con userId del dataset o ir a registrarse |
| `/create-user` | CreateUserPage | Buscar películas, asignar ratings, crear usuario |
| `/dashboard` | DashboardPage | Historial, recomendaciones, configuración del modelo, modal de explicación |

---

## Responsabilidades del equipo (5 personas)

| Persona | Punto del taller | Qué implementar en el backend | Funciones a conectar en `api.js` |
|---|---|---|---|
| **1** | Puntos 1 + 2 | Dataset, preprocesamiento, API de datos | `loginUser`, `getUserRatings`, `rateMovie`, `createUser`, `searchMovies` |
| **2** | Punto 3-i | Modelo User-User con **Jaccard** | `getRecommendations` cuando `similarity='jaccard'` |
| **3** | Puntos 3-ii/iii/e | Modelo User-User con **Cosine + Pearson + McLaughlin** | `getRecommendations` cuando `similarity='cosine'\|'pearson'`; `explainRecommendation` |
| **4** | Punto 4 | Modelo **Item-Item** (los 3 modelos) | `getRecommendations` cuando `modelType='item-item'`; `explainRecommendation` (vecinos de ítems) |
| **5** | Punto 5 | **Esta app web** | — integra todo |

---

## Cómo conectar el backend

Todo el tráfico pasa por **`src/api/api.js`**.
Cada función tiene un comentario `TODO` indicando el endpoint exacto.

1. Levanta tu servidor (ej. `http://localhost:8000`).
2. Descomenta/configura `BASE_URL` en `api.js`.
3. Reemplaza el cuerpo de cada función stub por la llamada `fetch`/`axios` real.
4. Mantener idénticas las firmas y formas de retorno — ninguna página necesita cambios.

---

## Contrato de API esperado

### Login

```
POST /api/login
Body:    { "userId": 1 }
Retorna: { "user": { "userId", "displayName", "totalRatings", "joinedAt" } }
         { "error": "..." }  si no existe
```

### Crear usuario

```
POST /api/users
Body:    { "ratings": [{ "movieId": 318, "rating": 5.0 }, ...] }
Retorna: { "user": { "userId", "displayName", "totalRatings", "joinedAt" } }
```

### Historial de ratings

```
GET /api/users/{userId}/ratings
Retorna: {
  "ratings": [
    { "movieId": 318, "title": "Shawshank Redemption, The (1994)",
      "genres": ["Crime","Drama"], "rating": 5.0, "ratedAt": "2000-07-30" },
    ...
  ]
}
```

### Agregar / actualizar rating

```
POST /api/users/{userId}/ratings
Body:    { "movieId": 79132, "rating": 4.5 }
Retorna: { "rating": { "movieId", "title", "genres", "rating", "ratedAt" } }
```

### Recomendaciones (con parámetros del modelo)

```
GET /api/users/{userId}/recommendations
    ?model=user-user          (o item-item)
    &similarity=pearson       (o jaccard, cosine)
    &neighbor_mode=k          (o threshold)
    &k=20
    &threshold=0.3
    &significance_weighting=false
    &significance_alpha=50
    &limit=10

Retorna: {
  "recommendations": [
    { "movieId": 79132, "title": "Inception (2010)",
      "genres": ["Action","Sci-Fi"], "predictedRating": 4.8, "rank": 1 },
    ...
  ]
}
```

### Explicación de una recomendación

```
GET /api/users/{userId}/recommendations/{movieId}/explain
    ?model=user-user&similarity=pearson&...  (mismos params del modelo)

Retorna: {
  "explanation": {
    "userId": 1,
    "movieId": 79132,
    "movieTitle": "Inception (2010)",
    "movieGenres": ["Action","Sci-Fi","Thriller"],
    "movieAvgRating": 4.18,
    "predictedRating": 4.8,
    "modelUsed": "User-User — Pearson, k=20",
    "similarityMetric": "pearson",
    "neighborsUsed": 20,

    "userRatingsEvidence": [
      { "movieId": 58559, "title": "Dark Knight, The (2008)",
        "genres": ["Action","Crime"], "rating": 5.0, "similarity": 0.91 },
      ...
    ],

    "neighborUsers": [                          ← solo en user-user
      {
        "userId": 3, "displayName": "Usuario 3",
        "similarity": 0.89, "ratingForMovie": 4.5,
        "sharedMovies": [
          { "movieId": 318, "title": "...", "userRating": 5.0, "neighborRating": 5.0 },
          ...
        ]
      },
      ...
    ],

    "neighborItems": [                          ← solo en item-item (Persona 4)
      { "movieId": 58559, "title": "Dark Knight, The (2008)",
        "genres": ["Action","Crime"], "avgRating": 4.3, "similarity": 0.87 },
      ...
    ]
  }
}
```

### Búsqueda de películas

```
GET /api/movies/search?q=inception
Retorna: {
  "movies": [
    { "movieId": 79132, "title": "Inception (2010)",
      "genres": ["Action","Sci-Fi","Thriller"], "avgRating": 4.18 },
    ...
  ]
}
Nota: la búsqueda local ya usa el catálogo real de movies.json (MovieLens 20M).
      Persona 1 puede reemplazar por búsqueda en el catálogo completo (~27.000 peliculas).
```

---

## Dataset

**MovieLens 20M** (GroupLens Research, University of Minnesota).
- `ratings.csv` → interacción usuario-ítem (20M valoraciones, escala 0.5–5.0 en pasos de 0.5)
- `movies.csv`  → catálogo de películas (movieId, título, géneros)
- El archivo `src/data/movies.json` contiene ~110 películas representativas del dataset real.

---

## Scripts disponibles

| Comando | Descripción |
|---|---|
| `npm start` | Dev server en `http://localhost:3000` |
| `npm run build` | Build de producción en `build/` |

---

## Stack

- React 18 · React Router v6
- React-Bootstrap 2 · Bootstrap 5
- Create React App (webpack)
- Diseño: sistema iOS (SF Pro, #F2F2F7, #007AFF)
