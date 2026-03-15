// api.js — Capa de integración con el backend Item-Item
// BASE_URL: 'http://localhost:8000/api'

const BASE_URL = 'http://localhost:8000/api';

// Helper centralizado para todas las peticiones al backend
async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// Construye la query string de parámetros del modelo
function buildModelQuery(params = {}) {
  const p = {
    model:                 params.modelType          ?? 'item-item',
    similarity:            params.similarity          ?? 'pearson',
    neighborMode:          params.neighborMode        ?? 'k',
    k:                     params.k                   ?? 20,
    threshold:             params.threshold           ?? 0.3,
    significanceWeighting: params.significanceWeighting ?? false,
    significanceAlpha:     params.significanceAlpha   ?? 50,
  };
  return new URLSearchParams(Object.entries(p).map(([k, v]) => [k, String(v)])).toString();
}

// POST /api/login  { userId }
export async function loginUser(userId) {
  const id = parseInt(userId, 10);
  if (isNaN(id) || id < 1) return { error: 'Ingresa un ID de usuario válido.' };
  try {
    return await apiFetch('/login', {
      method: 'POST',
      body: JSON.stringify({ userId: id }),
    });
  } catch (err) {
    return { error: err.message };
  }
}

// POST /api/users  { ratings: [...] }
export async function createUser(payload) {
  return apiFetch('/users', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

// GET /api/users/:id/ratings
export async function getUserRatings(userId) {
  return apiFetch(`/users/${userId}/ratings`);
}

// POST /api/users/:id/ratings  { movieId, rating }
export async function rateMovie(userId, movieId, rating) {
  return apiFetch(`/users/${userId}/ratings`, {
    method: 'POST',
    body: JSON.stringify({ movieId, rating }),
  });
}

// GET /api/users/:id/recommendations?model=item-item&similarity=...
export async function getRecommendations(userId, params = {}) {
  const qs = buildModelQuery(params) + `&limit=${params.limit ?? 10}`;
  return apiFetch(`/users/${userId}/recommendations?${qs}`);
}

// GET /api/users/:id/recommendations/:movieId/explain?...
export async function explainRecommendation(userId, movieId, params = {}) {
  const qs = buildModelQuery(params);
  return apiFetch(`/users/${userId}/recommendations/${movieId}/explain?${qs}`);
}

// GET /api/movies/search?q=...
export async function searchMovies(query) {
  if (!query || query.trim().length < 1) return { movies: [] };
  return apiFetch(`/movies/search?q=${encodeURIComponent(query.trim())}`);
}
