#!/usr/bin/env node
/**
 * build-data.js
 *
 * Procesa los CSV de MovieLens 20M en data/ y genera archivos JSON
 * que el frontend puede importar directamente (sin backend).
 *
 * Genera:
 *   frontend/src/data/movies-full.json   — catálogo completo de películas (~27K)
 *   frontend/src/data/sample-data.json   — usuarios reales + ratings + recs básicas
 *
 * Uso:
 *   node scripts/build-data.js
 */

const fs   = require('fs');
const path = require('path');
const rl   = require('readline');

const ROOT    = path.join(__dirname, '..');
const DATA    = path.join(ROOT, 'data');
const OUT_DIR = path.join(ROOT, 'frontend', 'src', 'data');

// Usuarios a extraer del dataset (IDs reales de MovieLens)
const SAMPLE_USERS = new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

// Máximo de ratings por usuario en el JSON final
const MAX_RATINGS_PER_USER = 150;

// Mínimo de ratings globales para aparecer en recomendaciones
const MIN_RATING_COUNT = 100;

// ─────────────────────────────────────────────────────────────────────────────

/** Lee un CSV línea por línea y llama a onRow([col1, col2, ...]). Salta la cabecera. */
function streamCSV(filePath, onRow) {
  return new Promise((resolve, reject) => {
    const iface = rl.createInterface({
      input: fs.createReadStream(filePath),
      crlfDelay: Infinity,
    });
    let firstLine = true;
    iface.on('line', (line) => {
      if (firstLine) { firstLine = false; return; }
      onRow(parseCSVLine(line));
    });
    iface.on('close', resolve);
    iface.on('error', reject);
  });
}

/** Parser CSV mínimo que respeta comillas. */
function parseCSVLine(line) {
  const out = [];
  let cur = '', inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"')                    { inQ = !inQ; }
    else if (c === ',' && !inQ)       { out.push(cur); cur = ''; }
    else                              { cur += c; }
  }
  out.push(cur);
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────

async function main() {

  // ── 1. Leer películas ────────────────────────────────────────────────
  process.stdout.write('Leyendo movie.csv... ');
  const movieMap = {};   // movieId → { movieId, title, genres[] }

  await streamCSV(path.join(DATA, 'movie.csv'), ([id, title, genres]) => {
    const mid = parseInt(id, 10);
    if (isNaN(mid)) return;
    movieMap[mid] = {
      movieId: mid,
      title: title ?? '',
      genres: (genres && genres !== '(no genres listed)')
        ? genres.split('|')
        : [],
    };
  });

  const moviesArray = Object.values(movieMap).sort((a, b) => a.movieId - b.movieId);
  console.log(`${moviesArray.length} películas`);

  // Escribir catálogo completo
  fs.writeFileSync(
    path.join(OUT_DIR, 'movies-full.json'),
    JSON.stringify(moviesArray),
  );
  console.log('  → movies-full.json escrito');

  // ── 2. Leer ratings ──────────────────────────────────────────────────
  process.stdout.write('Leyendo rating.csv (20M filas)...\n');

  const userRatingsRaw = {};   // userId → [{movieId, rating, ts}]  (sin enrich)
  const movieStats     = {};   // movieId → { sum, count }
  let totalRows = 0;

  await streamCSV(path.join(DATA, 'rating.csv'), ([uid, mid, rat, ts]) => {
    const userId  = parseInt(uid, 10);
    const movieId = parseInt(mid, 10);
    const rating  = parseFloat(rat);
    if (isNaN(userId) || isNaN(movieId) || isNaN(rating)) return;

    totalRows++;
    if (totalRows % 2_000_000 === 0) {
      process.stdout.write(`  ${(totalRows / 1_000_000).toFixed(0)}M filas...\n`);
    }

    // Estadísticas globales de cada película
    if (!movieStats[movieId]) movieStats[movieId] = { sum: 0, count: 0 };
    movieStats[movieId].sum   += rating;
    movieStats[movieId].count += 1;

    // Ratings de los usuarios de muestra
    if (SAMPLE_USERS.has(userId)) {
      if (!userRatingsRaw[userId]) userRatingsRaw[userId] = [];
      userRatingsRaw[userId].push({ movieId, rating, ts: parseInt(ts, 10) });
    }
  });

  console.log(`  ${totalRows.toLocaleString()} filas procesadas en total`);

  // ── 3. Enriquecer ratings con títulos y géneros ─────────────────────
  console.log('Construyendo ratings por usuario...');
  const userRatings = {};

  for (const [uid, rows] of Object.entries(userRatingsRaw)) {
    // Ordenar por timestamp descendente → más recientes primero
    rows.sort((a, b) => b.ts - a.ts);
    // Limitar cantidad
    const limited = rows.slice(0, MAX_RATINGS_PER_USER);
    userRatings[parseInt(uid, 10)] = limited.map(({ movieId, rating, ts }) => {
      const movie = movieMap[movieId];
      return {
        movieId,
        title:   movie?.title   ?? `Movie ${movieId}`,
        genres:  movie?.genres  ?? [],
        rating,
        ratedAt: new Date(ts * 1000).toISOString().slice(0, 10),
      };
    });
  }

  // ── 4. Construir lista de usuarios ───────────────────────────────────
  const users = Array.from(SAMPLE_USERS)
    .filter(uid => (userRatings[uid]?.length ?? 0) > 0)
    .map(uid => ({
      userId:       uid,
      displayName:  `Usuario ${uid}`,
      totalRatings: userRatings[uid].length,
      joinedAt:     '2000-01-01',
    }));

  console.log(`  Usuarios encontrados: ${users.map(u => u.userId).join(', ')}`);

  // ── 5. Construir recomendaciones básicas (popularidad real) ─────────
  // Para cada usuario: películas con alta puntuación global que no ha visto.
  console.log('Calculando recomendaciones basadas en popularidad...');

  // Pre-ordenar películas por avg rating desc (solo las con MIN_RATING_COUNT+)
  const globalTop = moviesArray
    .filter(m => (movieStats[m.movieId]?.count ?? 0) >= MIN_RATING_COUNT)
    .map(m => ({
      ...m,
      avgRating: movieStats[m.movieId].sum / movieStats[m.movieId].count,
      ratingCount: movieStats[m.movieId].count,
    }))
    .sort((a, b) => b.avgRating - a.avgRating || b.ratingCount - a.ratingCount);

  const recommendations = {};

  for (const uid of SAMPLE_USERS) {
    if (!userRatings[uid]) continue;
    const seen = new Set(userRatings[uid].map(r => r.movieId));

    recommendations[uid] = globalTop
      .filter(m => !seen.has(m.movieId))
      .slice(0, 20)
      .map((m, i) => ({
        movieId:         m.movieId,
        title:           m.title,
        genres:          m.genres,
        predictedRating: parseFloat(m.avgRating.toFixed(2)),
        rank:            i + 1,
      }));
  }

  // ── 6. Escribir sample-data.json ─────────────────────────────────────
  fs.writeFileSync(
    path.join(OUT_DIR, 'sample-data.json'),
    JSON.stringify({ users, ratings: userRatings, recommendations }),
  );
  console.log('  → sample-data.json escrito');
  console.log('\n¡Listo! Datos procesados correctamente.');
}

main().catch((err) => { console.error(err); process.exit(1); });
