import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Row, Col, Spinner } from 'react-bootstrap';
import { searchMovies, createUser } from '../api/api';
import RatingControl from '../components/RatingControl';

/**
 * CreateUserPage
 *
 * Registro de nuevo usuario: el usuario busca películas que ya ha visto,
 * asigna ratings y se registra en el sistema. Los ratings iniciales alimentan
 * el modelo de recomendación colaborativa.
 *
 * TODO (Persona 1): implementar POST /api/users en api.js → createUser()
 * TODO (Persona 1): reemplazar búsqueda local por GET /api/movies/search
 */
function CreateUserPage({ onLogin }) {
  const navigate = useNavigate();

  const [query,         setQuery]         = useState('');
  const [results,       setResults]       = useState([]);
  const [searching,     setSearching]     = useState(false);
  // { [movieId]: { movie, rating } }
  const [selected,      setSelected]      = useState({});
  const [creating,      setCreating]      = useState(false);
  const [error,         setError]         = useState('');
  const [successMsg,    setSuccessMsg]    = useState('');

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setResults([]);
    try {
      const { movies } = await searchMovies(query);
      setResults(movies || []);
    } catch { setError('Error al buscar. Intenta de nuevo.'); }
    finally { setSearching(false); }
  };

  const addMovie = (movie) => {
    if (selected[movie.movieId]) return;
    setSelected(prev => ({ ...prev, [movie.movieId]: { movie, rating: 3.5 } }));
  };

  const removeMovie = (id) => {
    setSelected(prev => { const n = { ...prev }; delete n[id]; return n; });
  };

  const handleRatingChange = (movieId, rating) => {
    setSelected(prev => ({ ...prev, [movieId]: { ...prev[movieId], rating: rating ?? 3.5 } }));
  };

  const handleCreate = async () => {
    setError('');
    const entries = Object.values(selected);
    if (entries.length < 3) {
      setError('Califica al menos 3 películas para generar recomendaciones.');
      return;
    }
    setCreating(true);
    try {
      const ratings = entries.map(({ movie, rating }) => ({ movieId: movie.movieId, rating }));
      const { user, error: err } = await createUser({ ratings });
      if (err) { setError(err); return; }
      setSuccessMsg(`Usuario creado. Tu ID es ${user.userId}. Redirigiendo...`);
      onLogin(user);
      setTimeout(() => navigate('/dashboard'), 1200);
    } catch { setError('No se pudo crear el usuario.'); }
    finally { setCreating(false); }
  };

  const selectedList = Object.values(selected);

  // ── Styles ──────────────────────────────────────────────────────
  const sectionTitle = { fontSize: '0.75rem', fontWeight: 600, color: '#8E8E93', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 10 };
  const card = { background: '#FFFFFF', borderRadius: 12, boxShadow: '0 1px 4px rgba(0,0,0,0.06)', padding: 20 };
  const listItem = (added) => ({
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '10px 0',
    borderBottom: '1px solid rgba(60,60,67,0.1)',
    background: added ? 'rgba(0,122,255,0.04)' : 'transparent',
  });
  const addBtn  = (added) => ({ padding: '5px 12px', borderRadius: 8, border: 'none', fontWeight: 500, fontSize: '0.8125rem', cursor: added ? 'default' : 'pointer', background: added ? '#E5E5EA' : '#007AFF', color: added ? '#8E8E93' : '#FFFFFF', whiteSpace: 'nowrap' });
  const removeBtn = { background: 'none', border: 'none', color: '#8E8E93', cursor: 'pointer', fontSize: '1rem', lineHeight: 1, padding: '0 2px' };

  return (
    <div style={{ minHeight: '100vh', background: '#F2F2F7', paddingBottom: 64 }}>
      {/* Nav */}
      <div style={{ background: 'rgba(255,255,255,0.85)', backdropFilter: 'blur(20px)', borderBottom: '1px solid rgba(60,60,67,0.12)', padding: '14px 0', position: 'sticky', top: 0, zIndex: 100 }}>
        <div style={{ maxWidth: 960, margin: '0 auto', padding: '0 20px', display: 'flex', alignItems: 'center', gap: 12 }}>
          <button onClick={() => navigate('/')} style={{ background: 'none', border: 'none', color: '#007AFF', fontSize: '0.9375rem', cursor: 'pointer', padding: 0, display: 'flex', alignItems: 'center', gap: 4 }}>
            &lsaquo; Volver
          </button>
          <span style={{ color: 'rgba(60,60,67,0.2)', fontSize: '1.2rem' }}>|</span>
          <span style={{ fontWeight: 600, fontSize: '1rem', color: '#1C1C1E' }}>Crear nuevo usuario</span>
        </div>
      </div>

      <div style={{ maxWidth: 960, margin: '0 auto', padding: '24px 20px' }}>
        {/* Header */}
        <h1 style={{ fontSize: '1.75rem', fontWeight: 700, color: '#1C1C1E', letterSpacing: '-0.02em', marginBottom: 6 }}>
          Nuevo usuario
        </h1>
        <p style={{ color: '#8E8E93', fontSize: '0.9375rem', marginBottom: 24 }}>
          Califica películas que ya hayas visto. Necesitas al menos <strong style={{ color: '#1C1C1E' }}>3 ratings</strong> para que el modelo genere recomendaciones.
        </p>

        {error && (
          <div style={{ background: 'rgba(255,59,48,0.06)', border: '1px solid rgba(255,59,48,0.2)', borderRadius: 10, padding: '10px 14px', color: '#FF3B30', fontSize: '0.875rem', marginBottom: 16 }}>
            {error}
          </div>
        )}
        {successMsg && (
          <div style={{ background: 'rgba(52,199,89,0.06)', border: '1px solid rgba(52,199,89,0.2)', borderRadius: 10, padding: '10px 14px', color: '#34C759', fontSize: '0.875rem', marginBottom: 16 }}>
            {successMsg}
          </div>
        )}

        <Row className="g-4">
          {/* Left — Search */}
          <Col md={7}>
            <div style={card}>
              <p style={sectionTitle}>Buscar películas</p>
              {/* TODO (Persona 1): búsqueda conectada a GET /api/movies/search?q={query} */}
              <form onSubmit={handleSearch} style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
                <input
                  type="text"
                  placeholder="Título, ej. Matrix, Inception..."
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  style={{ flex: 1, height: 40, borderRadius: 10, border: '1px solid rgba(60,60,67,0.18)', padding: '0 12px', fontSize: '0.9375rem', outline: 'none', color: '#1C1C1E' }}
                />
                <button type="submit" disabled={searching} style={{ height: 40, padding: '0 16px', borderRadius: 10, background: '#007AFF', color: '#FFF', border: 'none', fontWeight: 500, cursor: 'pointer', whiteSpace: 'nowrap', fontSize: '0.875rem' }}>
                  {searching ? <Spinner animation="border" size="sm" /> : 'Buscar'}
                </button>
              </form>

              {results.length === 0 && !searching && (
                <p style={{ color: '#C7C7CC', fontSize: '0.875rem', textAlign: 'center', padding: '20px 0' }}>
                  Los resultados aparecerán aquí
                </p>
              )}

              <div>
                {results.map(movie => {
                  const added = !!selected[movie.movieId];
                  return (
                    <div key={movie.movieId} style={listItem(added)}>
                      <div style={{ flex: 1, marginRight: 12 }}>
                        <div style={{ fontWeight: 500, fontSize: '0.9375rem', color: '#1C1C1E', lineHeight: 1.3 }}>
                          {movie.title}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#8E8E93', marginTop: 2 }}>
                          {movie.genres?.join(', ')}
                        </div>
                      </div>
                      <button style={addBtn(added)} onClick={() => addMovie(movie)} disabled={added}>
                        {added ? 'Agregada' : 'Agregar'}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          </Col>

          {/* Right — Selected ratings */}
          <Col md={5}>
            <div style={{ ...card, position: 'sticky', top: 80 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <p style={{ ...sectionTitle, marginBottom: 0 }}>Mis ratings</p>
                <span style={{ fontSize: '0.8125rem', color: selectedList.length >= 3 ? '#34C759' : '#8E8E93', fontWeight: 500 }}>
                  {selectedList.length} / min. 3
                </span>
              </div>

              {selectedList.length === 0 ? (
                <p style={{ color: '#C7C7CC', fontSize: '0.875rem', textAlign: 'center', padding: '24px 0' }}>
                  Agrega películas desde la búsqueda
                </p>
              ) : (
                <div>
                  {selectedList.map(({ movie, rating }) => (
                    <div key={movie.movieId} style={{ padding: '10px 0', borderBottom: '1px solid rgba(60,60,67,0.1)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
                        <span style={{ fontWeight: 500, fontSize: '0.875rem', color: '#1C1C1E', lineHeight: 1.3, flex: 1, marginRight: 8 }}>
                          {movie.title}
                        </span>
                        <button style={removeBtn} onClick={() => removeMovie(movie.movieId)} title="Quitar">&#x2715;</button>
                      </div>
                      <RatingControl
                        value={rating}
                        onChange={(r) => handleRatingChange(movie.movieId, r)}
                        size="sm"
                      />
                    </div>
                  ))}
                </div>
              )}

              <button
                onClick={handleCreate}
                disabled={creating || selectedList.length < 3}
                style={{ marginTop: 20, width: '100%', height: 44, borderRadius: 10, background: selectedList.length >= 3 ? '#007AFF' : '#E5E5EA', color: selectedList.length >= 3 ? '#FFF' : '#C7C7CC', border: 'none', fontSize: '0.9375rem', fontWeight: 600, cursor: selectedList.length >= 3 ? 'pointer' : 'not-allowed', transition: 'background 0.15s' }}
              >
                {creating ? 'Creando usuario...' : 'Crear usuario y ver recomendaciones'}
              </button>
            </div>
          </Col>
        </Row>
      </div>
    </div>
  );
}

export default CreateUserPage;
