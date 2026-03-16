import React, { useState } from 'react';
import { Modal, Row, Col } from 'react-bootstrap';
import RatingControl from './RatingControl';

function ExplanationModal({ show, onHide, explanation, loading, modelParams }) {
  const [tab, setTab] = useState('neighbors');

  if (!show) return null;

  const ratingColor = (r) => {
    if (!r) return '#8E8E93';
    if (r >= 4) return '#34C759';
    if (r >= 3) return '#FF9500';
    return '#FF3B30';
  };

  const similarityBar = (sim) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 4, background: '#F2F2F7', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ width: `${(sim * 100).toFixed(0)}%`, height: '100%', background: '#007AFF', borderRadius: 2 }} />
      </div>
      <span style={{ fontSize: '0.75rem', color: '#007AFF', fontWeight: 500, minWidth: '3ch', textAlign: 'right' }}>
        {(sim * 100).toFixed(0)}%
      </span>
    </div>
  );

  const isUserUser = explanation?.neighborUsers?.length > 0;
  const isItemItem = explanation?.neighborItems?.length > 0;

  const tabs = [
    ...(isUserUser ? [{ key: 'neighbors', label: 'Usuarios vecinos' }] : []),
    ...(isItemItem ? [{ key: 'items',     label: 'Items vecinos'    }] : []),
    { key: 'evidence', label: 'Tus valoraciones' },
  ];

  return (
    <Modal show={show} onHide={onHide} size="lg" centered>
      <Modal.Header closeButton style={{ background: '#FFFFFF', borderBottom: '1px solid rgba(60,60,67,0.12)', padding: '16px 20px' }}>
        <Modal.Title style={{ fontSize: '1.0625rem', fontWeight: 700, color: '#1C1C1E' }}>
          Por que se recomendog esta pelicula?
        </Modal.Title>
      </Modal.Header>

      <Modal.Body style={{ background: '#F2F2F7', padding: '20px' }}>
        {loading ? (
          <div style={{ textAlign: 'center', padding: '48px 0' }}>
            <div style={{ width: 28, height: 28, border: '2px solid #E5E5EA', borderTopColor: '#007AFF', borderRadius: '50%', animation: 'spin 0.8s linear infinite', margin: '0 auto 12px' }} />
            <p style={{ color: '#8E8E93', fontSize: '0.875rem', margin: 0 }}>Cargando explicacion...</p>
          </div>
        ) : !explanation ? (
          <p style={{ color: '#8E8E93', textAlign: 'center', padding: '48px 0', margin: 0 }}>
            No hay explicacion disponible.
          </p>
        ) : (
          <>
            {/* Movie summary card */}
            <div style={{ background: '#FFFFFF', borderRadius: 12, padding: '16px', marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>
              <Row className="align-items-center">
                <Col>
                  <h5 style={{ fontWeight: 700, color: '#1C1C1E', marginBottom: 6, fontSize: '1.0625rem' }}>
                    {explanation.movieTitle}
                  </h5>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
                    {explanation.movieGenres?.map(g => (
                      <span key={g} className="genre-tag">{g}</span>
                    ))}
                  </div>
                  <div style={{ display: 'flex', gap: 16, fontSize: '0.8125rem', color: '#8E8E93' }}>
                    <span>Rating promedio global: {explanation.movieAvgRating?.toFixed(2)}</span>
                    <span>Modelo: {explanation.modelUsed}</span>
                  </div>
                  {explanation.modelUsed && (
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                      {explanation.modelUsed.split(/[\s,]+/).filter(Boolean).map(t => (
                        <span key={t} style={{ padding: '2px 8px', background: '#F2F2F7', color: '#8E8E93', borderRadius: 5, fontSize: '0.75rem' }}>
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </Col>
                <Col xs="auto" style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '0.75rem', color: '#8E8E93', marginBottom: 4 }}>Rating predicho</div>
                  <div style={{ fontSize: '2.25rem', fontWeight: 800, color: ratingColor(explanation.predictedRating), lineHeight: 1 }}>
                    {explanation.predictedRating?.toFixed(1)}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#8E8E93' }}>de 5.0</div>
                </Col>
              </Row>
            </div>

            {/* Tabs */}
            <div className="seg-control" style={{ marginBottom: 12 }}>
              {tabs.map(t => (
                <button
                  key={t.key}
                  className={`seg-control-item${tab === t.key ? ' active' : ''}`}
                  onClick={() => setTab(t.key)}
                  type="button"
                >
                  {t.label}
                </button>
              ))}
            </div>

            {/* Tab: Neighbor users (user-user model) */}
            {tab === 'neighbors' && (
              <div>
                <p style={{ fontSize: '0.875rem', color: '#8E8E93', marginBottom: 12 }}>
                  Usuarios con perfil de gustos similar al tuyo que valoraron esta pelicula. Sus ratings se ponderaron por similitud para predecir tu valoracion.
                </p>
                {/* TODO: neighborUsers viene de GET /api/users/:id/recommendations/:movieId/explain */}
                {(explanation.neighborUsers || []).length === 0 && (
                  <p style={{ color: '#C7C7CC', textAlign: 'center', padding: '20px 0', fontSize: '0.875rem' }}>
                    Ninguno de tus usuarios similares calificó esta película directamente.
                  </p>
                )}
                {explanation.neighborUsers?.map(n => (
                  <div key={n.userId} style={{ background: '#FFFFFF', borderRadius: 10, padding: '14px 16px', marginBottom: 8, boxShadow: '0 1px 2px rgba(0,0,0,0.04)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                      <div>
                        <span style={{ fontWeight: 600, color: '#1C1C1E', fontSize: '0.9375rem' }}>{n.displayName}</span>
                        <span style={{ marginLeft: 8, fontSize: '0.8125rem', color: '#8E8E93' }}>Usuario {n.userId}</span>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <span style={{ fontWeight: 700, color: ratingColor(n.ratingForMovie), fontSize: '1.0625rem' }}>
                          {n.ratingForMovie?.toFixed(1)}
                        </span>
                        <span style={{ fontSize: '0.75rem', color: '#8E8E93', marginLeft: 4 }}>esta pelicula</span>
                      </div>
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      <div style={{ fontSize: '0.75rem', color: '#8E8E93', marginBottom: 4 }}>Similitud contigo</div>
                      {similarityBar(n.similarity)}
                    </div>
                    {n.sharedMovies?.length > 0 && (
                      <div>
                        <div style={{ fontSize: '0.75rem', color: '#8E8E93', marginBottom: 4 }}>Peliculas valoradas en comun</div>
                        {n.sharedMovies.map(sm => (
                          <div key={sm.movieId} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8125rem', padding: '3px 0', borderBottom: '1px solid rgba(60,60,67,0.06)' }}>
                            <span style={{ color: '#3C3C43' }}>{sm.title}</span>
                            <span style={{ color: '#8E8E93', whiteSpace: 'nowrap', marginLeft: 8 }}>
                              {sm.userRating != null ? `tu: ${sm.userRating}` : 'sin valorar'} / vecino: {sm.neighborRating}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Tab: Neighbor items (item-item model) */}
            {tab === 'items' && (
              <div>
                <p style={{ fontSize: '0.875rem', color: '#8E8E93', marginBottom: 12 }}>
                  Items similares a esta pelicula que el usuario ya valoro. La similitud item-item propaga sus ratings a esta prediccion.
                </p>
                {/* TODO : neighborItems viene de GET /api/users/:id/recommendations/:movieId/explain (item-item) */}
                {(explanation.neighborItems || []).length === 0 && (
                  <p style={{ color: '#C7C7CC', textAlign: 'center', padding: '20px 0', fontSize: '0.875rem' }}>
                    Sin datos de items vecinos — conectar backend (Persona 4)
                  </p>
                )}
                {explanation.neighborItems?.map(ni => (
                  <div key={ni.movieId} style={{ background: '#FFFFFF', borderRadius: 10, padding: '12px 16px', marginBottom: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
                      <span style={{ fontWeight: 500, color: '#1C1C1E' }}>{ni.title}</span>
                      <span style={{ color: '#8E8E93', fontSize: '0.8125rem', whiteSpace: 'nowrap', marginLeft: 8 }}>avg: {ni.avgRating?.toFixed(2)}</span>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#8E8E93', marginBottom: 4 }}>Similitud</div>
                    {similarityBar(ni.similarity)}
                  </div>
                ))}
              </div>
            )}

            {/* Tab: User's own rating evidence */}
            {tab === 'evidence' && (
              <div>
                <p style={{ fontSize: '0.875rem', color: '#8E8E93', marginBottom: 12 }}>
                  Tus valoraciones de peliculas similares que contribuyeron a esta prediccion. Mayor similitud = mayor influencia.
                </p>
                {/* TODO (Personas 2-4): userRatingsEvidence viene del endpoint de explicacion */}
                {(explanation.userRatingsEvidence || []).length === 0 && (
                  <p style={{ color: '#C7C7CC', textAlign: 'center', padding: '20px 0', fontSize: '0.875rem' }}>
                    Sin evidencia — conectar backend
                  </p>
                )}
                <div style={{ background: '#FFFFFF', borderRadius: 10, overflow: 'hidden', boxShadow: '0 1px 2px rgba(0,0,0,0.04)' }}>
                  {explanation.userRatingsEvidence?.map((item, i) => (
                    <div key={item.movieId} style={{ padding: '12px 16px', borderBottom: i < explanation.userRatingsEvidence.length - 1 ? '1px solid rgba(60,60,67,0.08)' : 'none' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                        <div style={{ flex: 1, minWidth: 0, marginRight: 12 }}>
                          <div style={{ fontWeight: 500, color: '#1C1C1E', fontSize: '0.9375rem' }}>{item.title}</div>
                          <div style={{ fontSize: '0.75rem', color: '#8E8E93', marginTop: 2 }}>
                            {item.genres?.join(', ')}
                          </div>
                        </div>
                        <RatingControl value={item.rating} readOnly size="sm" />
                      </div>
                      <div>
                        <div style={{ fontSize: '0.75rem', color: '#8E8E93', marginBottom: 3 }}>Similitud con la pelicula recomendada</div>
                        {similarityBar(item.similarity)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </Modal.Body>

      <Modal.Footer style={{ background: '#FFFFFF', borderTop: '1px solid rgba(60,60,67,0.12)', padding: '12px 20px' }}>
        <button
          onClick={onHide}
          style={{ height: 36, padding: '0 20px', borderRadius: 8, background: '#F2F2F7', border: 'none', color: '#1C1C1E', fontSize: '0.9375rem', fontWeight: 500, cursor: 'pointer' }}
        >
          Cerrar
        </button>
      </Modal.Footer>
    </Modal>
  );
}

export default ExplanationModal;
