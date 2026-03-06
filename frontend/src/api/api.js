// api.js — Capa de integración con el backend
// Conectar cada función con su endpoint real cuando el backend esté listo.
// BASE_URL: 'http://localhost:8000/api'

import MOVIES_FULL from '../data/movies-full.json';
import SAMPLE_DATA from '../data/sample-data.json';
import { MOCK_EXPLANATION } from '../mock/mockData';

const delay = (ms = 350) => new Promise((r) => setTimeout(r, ms));

export async function loginUser(userId) {
  await delay();
  const id = parseInt(userId, 10);
  if (isNaN(id) || id < 1) return { error: 'Ingresa un ID de usuario válido.' };
  // TODO Persona 2: POST /api/login  { userId }
  const user = SAMPLE_DATA.users.find((u) => u.userId === id);
  if (!user) {
    const valid = SAMPLE_DATA.users.map((u) => u.userId).join(', ');
    return { error: `Usuario ${id} no encontrado. IDs disponibles: ${valid}.` };
  }
  return { user };
}

export async function createUser(payload) {
  await delay(600);
  // TODO Persona 2: POST /api/users  { ratings: [...] }
  return {
    user: {
      userId:       Math.floor(Math.random() * 900000) + 100000,
      displayName:  'Nuevo Usuario',
      totalRatings: payload.ratings?.length ?? 0,
      joinedAt:     new Date().toISOString().slice(0, 10),
    },
  };
}

export async function getUserRatings(userId) {
  await delay();
  // TODO Persona 2: GET /api/users/:id/ratings
  return { ratings: SAMPLE_DATA.ratings[userId] ?? [] };
}

export async function rateMovie(userId, movieId, rating) {
  await delay(300);
  // TODO Persona 2: POST /api/users/:id/ratings  { movieId, rating }
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

// params: { modelType, similarity, neighborMode, k, threshold, significanceWeighting, significanceAlpha, limit }
// TODO Persona 3: user-user  → GET /api/users/:id/recommendations?model=user-user&similarity=...
// TODO Persona 4: item-item  → GET /api/users/:id/recommendations?model=item-item&similarity=...
export async function getRecommendations(userId, params = {}) {
  await delay(700);
  const limit = params.limit ?? 10;
  return { recommendations: (SAMPLE_DATA.recommendations[userId] ?? []).slice(0, limit) };
}

// TODO Persona 3/4: GET /api/users/:id/recommendations/:movieId/explain?model=...
export async function explainRecommendation(userId, movieId, params = {}) {
  await delay(600);
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

// TODO Persona 2: GET /api/movies/search?q=...  (catálogo completo ~27k películas)
export async function searchMovies(query) {
  await delay(150);
  if (!query || query.trim().length < 1) return { movies: [] };
  const q = query.toLowerCase().trim();
  return { movies: MOVIES_FULL.filter((m) => m.title.toLowerCase().includes(q)).slice(0, 20) };
}
