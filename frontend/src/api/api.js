/**
 * api.js — Integration layer
 *
 * ═══════════════════════════════════════════════════════════════════
 *  BACKEND TEAM: este es el ÚNICO archivo que deben modificar para
 *  conectar el backend real. Mantengan las firmas de las funciones
 *  idénticas para no romper ninguna página.
 * ═══════════════════════════════════════════════════════════════════
 *
 * RESPONSABILIDADES POR PERSONA
 * ─────────────────────────────────────────────────────────────────
 * Persona 1 — Puntos 1 + 2 (dataset y preprocesamiento)
 *   loginUser, getUserRatings, rateMovie, createUser, searchMovies
 *   Endpoints: POST /login, GET /users/:id/ratings,
 *              POST /users/:id/ratings, POST /users, GET /movies/search
 *
 * Persona 2 — Punto 3-i (User-User Jaccard)
 *   getRecommendations / explainRecommendation
 *   cuando params.modelType='user-user' && params.similarity='jaccard'
 *
 * Persona 3 — Puntos 3-ii/iii/e (User-User Cosine, Pearson, McLaughlin)
 *   getRecommendations / explainRecommendation
 *   cuando params.modelType='user-user' && params.similarity='cosine'|'pearson'
 *   Incluye significanceWeighting y significanceAlpha (ponderación de McLaughlin)
 *
 * Persona 4 — Punto 4 (Item-Item, todos los modelos)
 *   getRecommendations / explainRecommendation
 *   cuando params.modelType='item-item'
 *
 * Persona 5 — Punto 5 (Aplicación web) — este archivo ya está hecho.
 *
 * BASE URL — cambiar cuando el backend esté listo:
 *   const BASE_URL = 'http://localhost:8000/api';
 */

// Datos reales de MovieLens 20M generados por scripts/build-data.js
import MOVIES_FULL  from '../data/movies-full.json';
import SAMPLE_DATA  from '../data/sample-data.json';
import { MOCK_EXPLANATION } from '../mock/mockData';

const delay = (ms = 350) => new Promise((r) => setTimeout(r, ms));

// ── AUTH ─────────────────────────────────────────────────────────

/**
 * Autentica un usuario existente del dataset.
 *
 * Actualmente soporta los usuarios 1-10 del dataset real de MovieLens.
 * Cuando el backend esté listo reemplazar por:
 *
 * TODO (Persona 1): POST /api/login  { userId }
 * Respuesta:  { user: { userId, displayName, totalRatings, joinedAt } }
 *             { error: string }  si no existe
 */
export async function loginUser(userId) {
  await delay();
  const id = parseInt(userId, 10);
  if (isNaN(id) || id < 1) return { error: 'Ingresa un ID de usuario válido.' };
  // TODO: reemplazar por POST /api/login
  const user = SAMPLE_DATA.users.find((u) => u.userId === id);
  if (!user) {
    const valid = SAMPLE_DATA.users.map((u) => u.userId).join(', ');
    return { error: `Usuario ${id} no encontrado. IDs disponibles: ${valid}.` };
  }
  return { user };
}

// ── USER MANAGEMENT ──────────────────────────────────────────────

/**
 * Registra un nuevo usuario con ratings iniciales.
 *
 * TODO (Persona 1): POST /api/users
 * Body:     { ratings: [{ movieId, rating }] }
 * Respuesta: { user: { userId, displayName, totalRatings, joinedAt } }
 */
export async function createUser(payload) {
  await delay(600);
  // TODO: POST /api/users  { ratings: [...] }
  const newUser = {
    userId:       Math.floor(Math.random() * 900000) + 100000,
    displayName:  'Nuevo Usuario',
    totalRatings: payload.ratings?.length ?? 0,
    joinedAt:     new Date().toISOString().slice(0, 10),
  };
  return { user: newUser };
}

// ── RATINGS ──────────────────────────────────────────────────────

/**
 * Obtiene el historial de ratings de un usuario.
 * Usa los ratings reales de MovieLens para los usuarios 1-10.
 *
 * TODO (Persona 1): GET /api/users/:id/ratings
 * Respuesta: { ratings: [{ movieId, title, genres, rating, ratedAt }] }
 */
export async function getUserRatings(userId) {
  await delay();
  // TODO: GET /api/users/${userId}/ratings
  return { ratings: SAMPLE_DATA.ratings[userId] ?? [] };
}

/**
 * Agrega o actualiza un rating de un usuario para una película.
 *
 * TODO (Persona 1): POST /api/users/:id/ratings
 * Body:     { movieId, rating }
 * Respuesta: { rating: { movieId, title, genres, rating, ratedAt } }
 */
export async function rateMovie(userId, movieId, rating) {
  await delay(300);
  // TODO: POST /api/users/${userId}/ratings  { movieId, rating }
  const movie = MOVIES_FULL.find((m) => m.movieId === movieId);
  return {
    rating: {
      movieId,
      title:   movie?.title  ?? `Movie ${movieId}`,
      genres:  movie?.genres ?? [],
      rating,
      ratedAt: new Date().toISOString().slice(0, 10),
    },
  };
}

// ── RECOMMENDATIONS ──────────────────────────────────────────────

/**
 * Obtiene recomendaciones personalizadas para un usuario.
 *
 * Mientras el backend no esté listo, retorna las películas con mayor
 * puntuación promedio global (popularidad real de MovieLens 20M)
 * que el usuario no ha calificado.
 *
 * Los params del modelo corresponden directamente a los experimentos
 * del Taller 1 (puntos 3 y 4):
 *
 *   modelType            'user-user' | 'item-item'
 *   similarity           'jaccard' | 'cosine' | 'pearson'
 *   neighborMode         'k' | 'threshold'
 *   k                    entero  (cuando neighborMode === 'k')
 *   threshold            0–1     (cuando neighborMode === 'threshold')
 *   significanceWeighting boolean
 *   significanceAlpha    entero  (alpha de McLaughlin, si significanceWeighting === true)
 *   limit                entero  (máximo de recomendaciones, default 10)
 *
 * TODO (Persona 2): modelType='user-user', similarity='jaccard'
 *   GET /api/users/:id/recommendations?model=user-user&similarity=jaccard&...
 *
 * TODO (Persona 3): modelType='user-user', similarity='cosine'|'pearson'
 *   GET /api/users/:id/recommendations?model=user-user&similarity=cosine&...
 *   Incluir significance_weighting y significance_alpha como query params.
 *
 * TODO (Persona 4): modelType='item-item'
 *   GET /api/users/:id/recommendations?model=item-item&similarity=...&...
 *
 * Respuesta: { recommendations: [{ movieId, title, genres, predictedRating, rank }] }
 */
export async function getRecommendations(userId, params = {}) {
  await delay(700);
  // TODO: GET /api/users/${userId}/recommendations?${new URLSearchParams(params)}
  const limit = params.limit ?? 10;
  const all   = SAMPLE_DATA.recommendations[userId] ?? [];
  return { recommendations: all.slice(0, limit) };
}

/**
 * Obtiene la explicación de por qué se recomendó una película.
 * Pasar los mismos params usados en getRecommendations.
 *
 * TODO (Personas 2, 3, 4):
 *   GET /api/users/:id/recommendations/:movieId/explain?model=...&similarity=...
 *
 * Respuesta: {
 *   explanation: {
 *     userId, movieId, movieTitle, movieGenres, movieAvgRating,
 *     predictedRating, modelUsed, similarityMetric, neighborsUsed,
 *     userRatingsEvidence: [{ movieId, title, genres, rating, similarity }],
 *     neighborUsers (user-user): [{ userId, displayName, similarity, ratingForMovie, sharedMovies }],
 *     neighborItems (item-item): [{ movieId, title, genres, avgRating, similarity }]
 *   }
 * }
 */
export async function explainRecommendation(userId, movieId, params = {}) {
  await delay(600);
  // TODO: GET /api/users/${userId}/recommendations/${movieId}/explain?${new URLSearchParams(params)}
  const movie = MOVIES_FULL.find((m) => m.movieId === movieId);
  return {
    explanation: {
      ...MOCK_EXPLANATION,
      userId,
      movieId,
      movieTitle:  movie?.title  ?? MOCK_EXPLANATION.movieTitle,
      movieGenres: movie?.genres ?? MOCK_EXPLANATION.movieGenres,
    },
  };
}

// ── MOVIE SEARCH ─────────────────────────────────────────────────

/**
 * Busca películas por título en el catálogo completo de MovieLens 20M
 * (~27.000 películas, datos reales).
 *
 * TODO (Persona 1): reemplazar búsqueda local por
 *   GET /api/movies/search?q={query}
 *   Respuesta: { movies: [{ movieId, title, genres, avgRating }] }
 */
export async function searchMovies(query) {
  await delay(150);
  if (!query || query.trim().length < 1) return { movies: [] };
  const q = query.toLowerCase().trim();
  // TODO: GET /api/movies/search?q=${encodeURIComponent(query)}
  const movies = MOVIES_FULL
    .filter((m) => m.title.toLowerCase().includes(q))
    .slice(0, 20);
  return { movies };
}
