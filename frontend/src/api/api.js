// api.js — Conectado al backend FastAPI (main.py)
// Reemplaza src/api/api.js con este archivo completo.

const BASE_URL = "http://localhost:8000/api";

async function apiFetch(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Error en el servidor");
  return data;
}

// ── Login con usuario existente ─────────────────────────────────────────────
// El backend devuelve el usuario si existe en el dataset
export async function loginUser(userId) {
  const id = parseInt(userId, 10);
  if (isNaN(id) || id < 1) return { error: "Ingresa un ID de usuario válido." };
  try {
    const data = await apiFetch(`/users/${id}`);
    return { user: data.user };
  } catch (e) {
    return { error: e.message };
  }
}

// ── Crear nuevo usuario con sus preferencias (punto b) ──────────────────────
// ratings: [{movieId, rating}, ...]
export async function createUser(payload) {
  try {
    const data = await apiFetch("/users", {
      method: "POST",
      body: JSON.stringify({ ratings: payload.ratings ?? [] }),
    });
    return { user: data.user };
  } catch (e) {
    return { error: e.message };
  }
}

// ── Historial de ratings del usuario ────────────────────────────────────────
// Devuelve [{movieId, title, genres, rating, ratedAt}]
export async function getUserRatings(userId) {
  try {
    return await apiFetch(`/users/${userId}/ratings`);
  } catch (e) {
    return { ratings: [] };
  }
}

// ── Guardar un rating nuevo ──────────────────────────────────────────────────
export async function rateMovie(userId, movieId, rating) {
  try {
    return await apiFetch(`/users/${userId}/ratings`, {
      method: "POST",
      body: JSON.stringify({ movieId, rating }),
    });
  } catch (e) {
    return { error: e.message };
  }
}

// ── Recomendaciones del mejor modelo ────────────────────────────────────────
// El backend usa siempre el mejor modelo guardado en el pickle.
// params.limit controla cuántas recomendaciones devolver.
export async function getRecommendations(userId, params = {}) {
  const limit = params.limit ?? 10;
  try {
    return await apiFetch(`/users/${userId}/recommendations?limit=${limit}`);
  } catch (e) {
    return { recommendations: [] };
  }
}

// ── Explicación de por qué se recomendó una película (punto c) ──────────────
// Devuelve:
//   movieTitle, movieGenres, movieAvgRating   → info del ítem
//   predictedRating                           → rating que predijo el modelo
//   modelUsed                                 → ej: "User-User — pearson, k=20"
//   neighborUsers: [{                         → usuarios vecinos que influyeron
//     displayName, similarity,                →   nombre + % similitud
//     ratingForMovie,                         →   cómo calificó ESTA película
//     sharedMovies: [{title, userRating,      →   películas en común
//                     neighborRating}]
//   }]
export async function explainRecommendation(userId, movieId, params = {}) {
  try {
    return await apiFetch(
      `/users/${userId}/recommendations/${movieId}/explain`
    );
  } catch (e) {
    return { explanation: null };
  }
}

// ── Búsqueda de películas por título ────────────────────────────────────────
export async function searchMovies(query) {
  if (!query || query.trim().length < 1) return { movies: [] };
  try {
    return await apiFetch(`/movies/search?q=${encodeURIComponent(query)}`);
  } catch (e) {
    return { movies: [] };
  }
}

// Re-entrena el modelo con los ratings nuevos del usuario (~2-3 seg)
export async function refreshRecommendations(userId) {
  try {
    return await apiFetch(`/users/${userId}/refresh`, { method: "POST" });
  } catch (e) {
    return { error: e.message };
  }
}