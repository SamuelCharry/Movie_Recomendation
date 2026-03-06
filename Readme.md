# MovieLens Recommender — Frontend Shell

Workshop project for the **Systems of Recommendation** course.

This repository contains **only the React frontend**. It is a visual scaffold ready to be connected to a real backend and recommendation model by the team.

---

## Quick Start

```bash
cd frontend
npm install
npm start
# http://localhost:3000
```

Demo user IDs: **1** (Alice), **2** (Bob), **3** (Carol), **4** (David).

---

## What is in this repo

```
frontend/
  src/
    api/api.js          ← stub functions, replace with real backend calls
    mock/mockData.js    ← temporary mock data (MovieLens-like)
    components/
      ExplanationModal.jsx
    pages/
      LoginPage.jsx
      CreateUserPage.jsx
      DashboardPage.jsx
    App.jsx
    index.js
  package.json
  README.md             ← full documentation + backend API contract
```

See `frontend/README.md` for the complete integration guide and backend API contract.

---

## Dataset

**MovieLens 20M Dataset** (GroupLens Research). User-item interactions from `ratings.csv`.

---

## What was removed

The original repository was a multi-language monorepo (Express + MongoDB + Django + scikit-learn).
The following were deleted during refactoring:

- `Backend/` — Node.js/Express/MongoDB server
- `ML_Model/` — Jupyter notebooks, CSV files
- `Movie_Python/` — Django REST API, SQLite database, pickle files
- Root `package.json` concurrently scripts
- All TMDB references, old Navbar, ProjectDetails page, test boilerplate

Only frontend code remains.
