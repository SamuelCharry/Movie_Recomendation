import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Row, Col, Form, Button, Alert, Spinner } from 'react-bootstrap';
import { loginUser } from '../api/api';

const S = {
  page: {
    minHeight: '100vh',
    background: '#F2F2F7',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '24px 16px',
  },
  card: {
    background: '#FFFFFF',
    borderRadius: 16,
    boxShadow: '0 4px 24px rgba(0,0,0,0.06)',
    padding: '36px 32px',
    width: '100%',
    maxWidth: 400,
  },
  appTitle: {
    fontSize: '1.625rem',
    fontWeight: 700,
    color: '#1C1C1E',
    letterSpacing: '-0.02em',
    marginBottom: 4,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: '0.875rem',
    color: '#8E8E93',
    textAlign: 'center',
    marginBottom: 32,
  },
  sectionLabel: {
    display: 'block',
    fontSize: '0.75rem',
    fontWeight: 600,
    color: '#8E8E93',
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
    marginBottom: 6,
  },
  input: {
    height: 44,
    borderRadius: 10,
    border: '1px solid rgba(60,60,67,0.18)',
    fontSize: '0.9375rem',
    padding: '0 12px',
    color: '#1C1C1E',
    background: '#FFFFFF',
    width: '100%',
    outline: 'none',
    transition: 'border-color 0.15s',
  },
  primaryBtn: {
    width: '100%',
    height: 44,
    borderRadius: 10,
    background: '#007AFF',
    color: '#FFFFFF',
    border: 'none',
    fontSize: '0.9375rem',
    fontWeight: 600,
    cursor: 'pointer',
    marginTop: 8,
  },
  secondaryBtn: {
    width: '100%',
    height: 44,
    borderRadius: 10,
    background: '#F2F2F7',
    color: '#007AFF',
    border: 'none',
    fontSize: '0.9375rem',
    fontWeight: 500,
    cursor: 'pointer',
    marginTop: 8,
  },
  divider: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    margin: '16px 0',
    color: '#C7C7CC',
    fontSize: '0.8125rem',
  },
  dividerLine: {
    flex: 1,
    height: 1,
    background: 'rgba(60,60,67,0.12)',
  },
  hint: {
    fontSize: '0.75rem',
    color: '#8E8E93',
    marginTop: 6,
  },
};

/**
 * LoginPage
 *
 * Permite a un usuario existente del dataset autenticarse con su userId,
 * o ir al flujo de creación de nuevo usuario.
 *
 * TODO (Persona 1): implementar POST /api/login en api.js → loginUser()
 *   Una vez conectado, el backend puede retornar un token JWT para manejo
 *   de sesión real. Guardar el token aquí y pasarlo en cada request.
 */
function LoginPage({ onLogin }) {
  const [userId, setUserId]   = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!userId.trim()) { setError('Ingresa tu ID de usuario.'); return; }
    setLoading(true);
    try {
      const { user, error: err } = await loginUser(userId);
      if (err) { setError(err); return; }
      onLogin(user);
      navigate('/dashboard');
    } catch {
      setError('No se pudo conectar con el servidor.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={S.page}>
      <div style={S.card}>
        <h1 style={S.appTitle}>MovieLens Recommender</h1>
        <p style={S.subtitle}>
          Sistema de recomendación colaborativa · MovieLens 20M
        </p>

        {error && (
          <div style={{
            background: 'rgba(255,59,48,0.08)',
            border: '1px solid rgba(255,59,48,0.2)',
            borderRadius: 10,
            padding: '10px 14px',
            fontSize: '0.875rem',
            color: '#FF3B30',
            marginBottom: 16,
          }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <label style={S.sectionLabel}>ID de Usuario</label>
          <input
            type="number"
            min="1"
            placeholder="Ej. 1, 2, 3 o 4"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            style={S.input}
            onFocus={e => (e.target.style.borderColor = '#007AFF')}
            onBlur={e  => (e.target.style.borderColor = 'rgba(60,60,67,0.18)')}
          />
          <p style={S.hint}>
            Usuarios disponibles (datos reales MovieLens 20M): 1 al 10
          </p>

          <button type="submit" style={S.primaryBtn} disabled={loading}>
            {loading ? 'Iniciando sesión...' : 'Iniciar sesión'}
          </button>
        </form>

        <div style={S.divider}>
          <span style={S.dividerLine} />
          <span>o</span>
          <span style={S.dividerLine} />
        </div>

        <button
          type="button"
          style={S.secondaryBtn}
          onClick={() => navigate('/create-user')}
        >
          Crear nuevo usuario
        </button>
      </div>
    </div>
  );
}

export default LoginPage;
