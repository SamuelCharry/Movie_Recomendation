import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LoginPage      from './pages/LoginPage';
import CreateUserPage from './pages/CreateUserPage';
import DashboardPage  from './pages/DashboardPage';

// Parametros del modelo de recomendacion (Puntos 3 y 4)
const DEFAULT_MODEL_PARAMS = {
  modelType:             'user-user',
  similarity:            'pearson',
  neighborMode:          'k',
  k:                     20,
  threshold:             0.3,
  significanceWeighting: false,
  significanceAlpha:     50,
};

function App() {
  const [user,        setUser]        = useState(null);
  const [modelParams, setModelParams] = useState(DEFAULT_MODEL_PARAMS);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LoginPage onLogin={setUser} />} />
        <Route path="/create-user" element={<CreateUserPage onLogin={setUser} />} />
        <Route
          path="/dashboard"
          element={
            user ? (
              <DashboardPage
                user={user}
                onLogout={() => setUser(null)}
                modelParams={modelParams}
                onModelParamsChange={setModelParams}
              />
            ) : (
              <Navigate to="/" replace />
            )
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
