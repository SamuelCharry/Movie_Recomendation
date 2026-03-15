import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Row, Col, Spinner } from 'react-bootstrap';
import { getUserRatings, getRecommendations, explainRecommendation, rateMovie,refreshRecommendations} from '../api/api';
import RatingControl from '../components/RatingControl';
import ExplanationModal from '../components/ExplanationModal';


// ── Segmented control helper ─────────────────────────────────────
function Seg({ options, value, onChange }) {
  return (
    <div className="seg-control">
      {options.map(opt => (
        <button
          key={opt.value}
          className={`seg-control-item${value === opt.value ? ' active' : ''}`}
          onClick={() => onChange(opt.value)}
          type="button"
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}


const ratingColor = (r) => {
  if (r >= 4) return '#34C759';
  if (r >= 3) return '#FF9500';
  return '#FF3B30';
};

/**
 * DashboardPage — Vista principal del usuario autenticado.
 */
function DashboardPage({ user, onLogout, modelParams, onModelParamsChange }) {
  const navigate = useNavigate();

  // Data
  const [ratings,          setRatings]          = useState([]);
  const [recommendations,  setRecommendations]  = useState([]);
  const [ratingsLoading,   setRatingsLoading]   = useState(true);
  const [recsLoading,      setRecsLoading]      = useState(true);
  const [error,            setError]            = useState('');

  // Model config panel
  const [configOpen,   setConfigOpen]   = useState(false);
  const [localParams,  setLocalParams]  = useState(modelParams);

  // Inline rating
  const [ratingMovieId, setRatingMovieId] = useState(null);
  const [inlineRating,  setInlineRating]  = useState(null);
  const [savingRating,  setSavingRating]  = useState(false);

  // Explanation modal
  const [showExpl,   setShowExpl]   = useState(false);
  const [expl,       setExpl]       = useState(null);
  const [explLoading, setExplLoading] = useState(false);
  const [explMovie,   setExplMovie]   = useState(null);

  useEffect(() => { if (!user) navigate('/'); }, [user, navigate]);

  const loadData = (params = modelParams) => {
    if (!user) return;
    setRatingsLoading(true);
    setRecsLoading(true);
    setError('');
    getUserRatings(user.userId)
      .then(r => setRatings(r.ratings || []))
      .catch(() => setError('Error al cargar ratings.'))
      .finally(() => setRatingsLoading(false));
    getRecommendations(user.userId, { ...params, limit: 10 })
      .then(r => setRecommendations(r.recommendations || []))
      .catch(() => setError('Error al cargar recomendaciones.'))
      .finally(() => setRecsLoading(false));
  };

  useEffect(() => { loadData(); }, [user]); // eslint-disable-line

  const applyParams = () => {
    onModelParamsChange(localParams);
    loadData(localParams);
    setConfigOpen(false);
  };

  const handleExplain = async (rec) => {
    setExplMovie(rec);
    setShowExpl(true);
    setExpl(null);
    setExplLoading(true);
    try {
      const { explanation } = await explainRecommendation(user.userId, rec.movieId, modelParams);
      setExpl(explanation);
    } catch {
      setError('Error al cargar la explicación.');
    } finally {
      setExplLoading(false);
    }
  };

  const handleRateConfirm = async (rec) => {
    if (inlineRating == null) return;
    setSavingRating(true);
    try {
      const { rating: newEntry } = await rateMovie(user.userId, rec.movieId, inlineRating);
      // Agrega al historial
      setRatings(prev => [newEntry, ...prev.filter(r => r.movieId !== rec.movieId)]);
      // Recarga recomendaciones para mantener siempre 10
      const { recommendations: newRecs } = await getRecommendations(user.userId, { ...modelParams, limit: 10 });
      setRecommendations(newRecs || []);
    } finally {
      setSavingRating(false);
      setRatingMovieId(null);
      setInlineRating(null);
    }
  };

  const handleLogout = () => { onLogout(); navigate('/'); };

  if (!user) return null;

  // ── Styles ────────────────────────────────────────────────────
  const section = { marginBottom: 32 };
  const sectionLabel = { fontSize: '0.6875rem', fontWeight: 600, color: '#8E8E93', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 };
  const card = { background: '#FFFFFF', borderRadius: 12, boxShadow: '0 1px 3px rgba(0,0,0,0.06)', overflow: 'hidden' };
  const listRow = { padding: '12px 16px', borderBottom: '1px solid rgba(60,60,67,0.08)' };
  const configRow = { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid rgba(60,60,67,0.08)' };
  const configLabel = { fontSize: '0.9375rem', color: '#1C1C1E' };
  const configDesc  = { fontSize: '0.8125rem', color: '#8E8E93', marginTop: 2 };

  return (
    <div style={{ minHeight: '100vh', background: '#F2F2F7', paddingBottom: 64 }}>

      {/* ── Nav bar ─────────────────────────────────────────────── */}
      <div style={{ background: 'rgba(255,255,255,0.85)', backdropFilter: 'blur(20px)', borderBottom: '1px solid rgba(60,60,67,0.12)', padding: '0', position: 'sticky', top: 0, zIndex: 100 }}>
        <div style={{ maxWidth: 1100, margin: '0 auto', padding: '12px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontWeight: 700, fontSize: '1.0625rem', color: '#1C1C1E', letterSpacing: '-0.01em' }}>
            MovieLens Recommender
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <span style={{ fontSize: '0.875rem', color: '#8E8E93' }}>
              {user.displayName} · ID {user.userId}
            </span>
            <button
              onClick={handleLogout}
              style={{ background: 'none', border: 'none', color: '#FF3B30', fontSize: '0.875rem', cursor: 'pointer', fontWeight: 500 }}
            >
              Salir
            </button>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '28px 20px' }}>

        {error && (
          <div style={{ background: 'rgba(255,59,48,0.06)', border: '1px solid rgba(255,59,48,0.2)', borderRadius: 10, padding: '10px 14px', color: '#FF3B30', fontSize: '0.875rem', marginBottom: 20 }}>
            {error}
            <button onClick={() => setError('')} style={{ background: 'none', border: 'none', color: '#FF3B30', float: 'right', cursor: 'pointer', fontWeight: 700 }}>&#x2715;</button>
          </div>
        )}

        {/* ── User summary ─────────────────────────────────────── */}
        <div style={{ marginBottom: 28 }}>
          <h1 style={{ fontSize: '2rem', fontWeight: 700, color: '#1C1C1E', letterSpacing: '-0.03em', marginBottom: 4 }}>
            {user.displayName}
          </h1>
          <p style={{ color: '#8E8E93', fontSize: '0.9375rem', marginBottom: 0 }}>
            {user.totalRatings} valoraciones · Miembro desde {user.joinedAt}
          </p>
        </div>

        {/* ── Model configuration ─────────────────────────────── */}
        <div style={{ marginBottom: 28 }}>
          <p style={sectionLabel}>Modelo activo</p>
          <div style={{ ...card, padding: '12px 16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
 
              {/* Info fija del mejor modelo */}
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                <span style={{ fontSize: '0.875rem', color: '#8E8E93' }}>Mejor modelo según experimentos:</span>
                {[
                  'pearson-baseline',
                  'k_vecinos=50',
                  'umbral_similitud=5',
                  'gamma=200',
                  'MAE=0.588',
                  'RMSE=0.771',
                ].map(tag => (
                  <span key={tag} style={{
                    display: 'inline-block', padding: '3px 10px', borderRadius: 6,
                    background: 'rgba(52,199,89,0.1)', color: '#34C759',
                    fontSize: '0.8125rem', fontWeight: 500,
                  }}>
                    {tag}
                  </span>
                ))}
              </div>
 
              {/* Toggle User-User / Item-Item */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontSize: '0.875rem', color: '#1C1C1E', fontWeight: 500 }}>Tipo:</span>
                <Seg
                  value={modelParams.modelType}
                  onChange={v => {
                    onModelParamsChange({ ...modelParams, modelType: v });
                    loadData({ ...modelParams, modelType: v });
                  }}
                  options={[
                    { label: 'User-User', value: 'user-user' },
                    { label: 'Item-Item', value: 'item-item' },
                  ]}
                />
              </div>
 
            </div>
          </div>
        </div>

        {/* ── Main content ─────────────────────────────────────── */}
        <Row className="g-4">

          {/* LEFT — Rating history */}
          <Col lg={4}>
            <div style={section}>
              <p style={sectionLabel}>Historial de valoraciones</p>
              <div style={card}>
                {ratingsLoading ? (
                  <div style={{ padding: 32, textAlign: 'center' }}>
                    <Spinner animation="border" size="sm" style={{ color: '#007AFF' }} />
                    <p style={{ color: '#8E8E93', fontSize: '0.875rem', marginTop: 8, marginBottom: 0 }}>Cargando...</p>
                  </div>
                ) : ratings.length === 0 ? (
                  <div style={{ padding: 32, textAlign: 'center', color: '#C7C7CC', fontSize: '0.875rem' }}>
                    Sin valoraciones aún
                  </div>
                ) : (
                  <div style={{ maxHeight: 560, overflowY: 'auto' }}>
                    {ratings.map((item, i) => (
                      <div key={`${item.movieId}-${i}`} style={{ ...listRow, ...(i === ratings.length - 1 ? { borderBottom: 'none' } : {}) }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <div style={{ flex: 1, marginRight: 12 }}>
                            <div style={{ fontWeight: 500, fontSize: '0.9rem', color: '#1C1C1E', lineHeight: 1.3 }}>
                              {item.title}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: '#8E8E93', marginTop: 2 }}>
                              {item.genres?.slice(0, 3).join(', ')} · {item.ratedAt}
                            </div>
                          </div>
                          <RatingControl value={item.rating} readOnly size="sm" />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </Col>

          {/* RIGHT — Recommendations */}
          <Col lg={8}>
            <div style={section}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                <p style={{ ...sectionLabel, marginBottom: 0 }}>Recomendaciones para ti</p>
                <button
                  onClick={async () => {
                    setRecsLoading(true);
                    await refreshRecommendations(user.userId);
                    const { recommendations: newRecs } = await getRecommendations(user.userId, { ...modelParams, limit: 10 });
                    setRecommendations(newRecs || []);
                    setRecsLoading(false);
                  }}
                  disabled={recsLoading}
                  style={{ height: 30, padding: '0 12px', borderRadius: 8, background: 'rgba(0,122,255,0.08)', border: 'none', fontSize: '0.8125rem', fontWeight: 500, color: '#007AFF', cursor: 'pointer' }}
                >
                  {recsLoading ? 'Actualizando...' : '↻ Actualizar'}
                </button>
              </div>

              {recsLoading ? (
                <div style={{ ...card, padding: 48, textAlign: 'center' }}>
                  <Spinner animation="border" size="sm" style={{ color: '#007AFF' }} />
                  <p style={{ color: '#8E8E93', fontSize: '0.875rem', marginTop: 8, marginBottom: 0 }}>Generando recomendaciones...</p>
                </div>
              ) : recommendations.length === 0 ? (
                <div style={{ ...card, padding: 48, textAlign: 'center', color: '#C7C7CC', fontSize: '0.875rem' }}>
                  Sin recomendaciones disponibles
                </div>
              ) : (
                <div style={card}>
                  {recommendations.map((rec, idx) => (
                    <div key={rec.movieId} style={{ ...listRow, ...(idx === recommendations.length - 1 ? { borderBottom: 'none' } : {}) }}>
                      {/* Header row */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                        {/* Rank */}
                        <div style={{
                          width: 30, height: 30, borderRadius: '50%', flexShrink: 0,
                          background: idx === 0 ? '#007AFF' : '#F2F2F7',
                          color: idx === 0 ? '#FFF' : '#8E8E93',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          fontSize: '0.8125rem', fontWeight: 700,
                        }}>
                          {rec.rank}
                        </div>

                        {/* Title and genres */}
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontWeight: 600, fontSize: '0.9375rem', color: '#1C1C1E', lineHeight: 1.3 }}>
                            {rec.title}
                          </div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
                            {rec.genres?.slice(0, 4).map(g => (
                              <span key={g} className="genre-tag">{g}</span>
                            ))}
                          </div>
                        </div>

                        {/* Predicted rating */}
                        <div style={{ textAlign: 'right', flexShrink: 0 }}>
                          <div style={{ fontSize: '1.375rem', fontWeight: 800, color: ratingColor(rec.predictedRating), lineHeight: 1 }}>
                            {rec.predictedRating?.toFixed(1)}
                          </div>
                          <div style={{ fontSize: '0.6875rem', color: '#8E8E93', marginTop: 2 }}>predicho</div>
                        </div>

                        {/* Actions */}
                        <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
                          <button
                            onClick={() => { setRatingMovieId(ratingMovieId === rec.movieId ? null : rec.movieId); setInlineRating(null); }}
                            style={{ height: 32, padding: '0 12px', borderRadius: 8, background: '#F2F2F7', border: 'none', fontSize: '0.8125rem', fontWeight: 500, color: '#1C1C1E', cursor: 'pointer' }}
                          >
                            Valorar
                          </button>
                          <button
                            onClick={() => handleExplain(rec)}
                            style={{ height: 32, padding: '0 12px', borderRadius: 8, background: 'rgba(0,122,255,0.08)', border: 'none', fontSize: '0.8125rem', fontWeight: 500, color: '#007AFF', cursor: 'pointer' }}
                          >
                            Por que?
                          </button>
                        </div>
                      </div>

                      {/* Inline rating panel */}
                      {ratingMovieId === rec.movieId && (
                        <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid rgba(60,60,67,0.08)', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                          <span style={{ fontSize: '0.875rem', color: '#8E8E93' }}>Tu valoracion:</span>
                          <RatingControl
                            value={inlineRating}
                            onChange={setInlineRating}
                          />
                          <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
                            <button
                              onClick={() => { setRatingMovieId(null); setInlineRating(null); }}
                              style={{ height: 32, padding: '0 12px', borderRadius: 8, background: '#F2F2F7', border: 'none', fontSize: '0.8125rem', color: '#1C1C1E', cursor: 'pointer' }}
                            >
                              Cancelar
                            </button>
                            <button
                              onClick={() => handleRateConfirm(rec)}
                              disabled={inlineRating == null || savingRating}
                              style={{ height: 32, padding: '0 14px', borderRadius: 8, background: inlineRating != null ? '#007AFF' : '#E5E5EA', border: 'none', fontSize: '0.8125rem', fontWeight: 600, color: inlineRating != null ? '#FFF' : '#C7C7CC', cursor: inlineRating != null ? 'pointer' : 'not-allowed' }}
                            >
                              {savingRating ? 'Guardando...' : 'Confirmar'}
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Col>
        </Row>
      </div>

      {/* Explanation modal */}
      <ExplanationModal
        show={showExpl}
        onHide={() => { setShowExpl(false); setExpl(null); }}
        explanation={expl}
        loading={explLoading}
        movie={explMovie}
        modelParams={modelParams}
      />
    </div>
  );
}

export default DashboardPage;
