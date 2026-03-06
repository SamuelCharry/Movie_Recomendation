/**
 * mockData.js — Datos temporales para explicaciones
 *
 * Solo se usa para el endpoint de explicación mientras el backend
 * no está conectado. Los usuarios, ratings y recomendaciones ahora
 * vienen de los datos reales de MovieLens 20M (sample-data.json).
 *
 * TODO (Personas 2, 3, 4): reemplazar MOCK_EXPLANATION conectando
 *   GET /api/users/:id/recommendations/:movieId/explain en api.js
 */

// ── EXPLICACIÓN DE MUESTRA ────────────────────────────────────────
export const MOCK_EXPLANATION = {
  userId: 1,
  movieId: 79132,
  movieTitle: 'Inception (2010)',
  movieGenres: ['Action', 'Crime', 'Drama', 'Mystery', 'Sci-Fi', 'Thriller'],
  movieAvgRating: 4.18,
  predictedRating: 4.8,
  modelUsed: 'User-User — Pearson, k=20',
  similarityMetric: 'pearson',
  neighborsUsed: 20,

  // Películas que el usuario ya calificó y son similares a la recomendación
  userRatingsEvidence: [
    { movieId: 58559, title: 'Dark Knight, The (2008)',           genres: ['Action','Crime','Drama'], rating: 5.0, similarity: 0.91 },
    { movieId: 2959,  title: 'Fight Club (1999)',                 genres: ['Action','Crime','Drama'], rating: 4.0, similarity: 0.87 },
    { movieId: 296,   title: 'Pulp Fiction (1994)',               genres: ['Crime','Drama'],          rating: 4.5, similarity: 0.84 },
    { movieId: 318,   title: 'Shawshank Redemption, The (1994)', genres: ['Crime','Drama'],          rating: 5.0, similarity: 0.82 },
  ],

  // Vecinos (modelo user-user) — datos de Personas 2 y 3
  neighborUsers: [
    {
      userId: 3,
      displayName: 'Usuario 3',
      similarity: 0.89,
      ratingForMovie: 4.5,
      sharedMovies: [
        { movieId: 318,   title: 'Shawshank Redemption, The (1994)', userRating: 5.0, neighborRating: 5.0 },
        { movieId: 2959,  title: 'Fight Club (1999)',                 userRating: 4.0, neighborRating: 3.5 },
        { movieId: 58559, title: 'Dark Knight, The (2008)',           userRating: 5.0, neighborRating: 4.5 },
      ],
    },
    {
      userId: 2,
      displayName: 'Usuario 2',
      similarity: 0.74,
      ratingForMovie: 4.0,
      sharedMovies: [
        { movieId: 480,  title: 'Jurassic Park (1993)',                      userRating: null, neighborRating: 3.5 },
        { movieId: 4993, title: 'Lord of the Rings: The Fellowship… (2001)', userRating: null, neighborRating: 5.0 },
      ],
    },
  ],

  // Ítems vecinos (modelo item-item) — completar por Persona 4
  // TODO (Persona 4): poblar neighborItems desde el modelo item-item
  neighborItems: [],
};
