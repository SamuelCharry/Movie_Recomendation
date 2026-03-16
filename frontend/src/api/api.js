// api.js — Conectado al backend FastAPI (main.py)
// BASE_URL: http://localhost:8000/api

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

// ── Login con usuario existente ──────────────────────────────────────────────
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

// ── Crear nuevo usuario con sus preferencias ─────────────────────────────────
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

// ── Historial de ratings del usuario ─────────────────────────────────────────
export async function getUserRatings(userId) {
  try {
    return await apiFetch(`/users/${userId}/ratings`);
  } catch (e) {
    return { ratings: [] };
  }
}

// ── Guardar un rating nuevo ───────────────────────────────────────────────────
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

// ── Recomendaciones ───────────────────────────────────────────────────────────
// params.modelType: "user-user" o "item-item"
// params.limit: número de recomendaciones
export async function getRecommendations(userId, params = {}) {
  const limit = params.limit ?? 10;
  const model = params.modelType ?? "user-user";
  try {
    return await apiFetch(
      `/users/${userId}/recommendations?limit=${limit}&model=${model}`
    );
  } catch (e) {
    return { recommendations: [] };
  }
}

// ── Explicación de por qué se recomendó una película ─────────────────────────
// User-User → muestra usuarios vecinos similares y películas en común
// Item-Item → muestra películas similares del historial del usuario
export async function explainRecommendation(userId, movieId, params = {}) {
  const model = params.modelType ?? "user-user";
  try {
    return await apiFetch(
      `/users/${userId}/recommendations/${movieId}/explain?model=${model}`
    );
  } catch (e) {
    return { explanation: null };
  }
}

// ── Búsqueda de películas por título ─────────────────────────────────────────
export async function searchMovies(query) {
  if (!query || query.trim().length < 1) return { movies: [] };
  try {
    return await apiFetch(`/movies/search?q=${encodeURIComponent(query)}`);
  } catch (e) {
    return { movies: [] };
  }
}

// ── Re-entrena User-User con ratings nuevos del usuario (~2-3 seg) ────────────
export async function refreshRecommendations(userId) {
  try {
    return await apiFetch(`/users/${userId}/refresh`, { method: "POST" });
  } catch (e) {
    return { error: e.message };
  }
}