# Guía de Ejecución — Sistema de Recomendación de Películas

> **Stack:** Python 3.9+ · FastAPI · React 18
> **Modelos:** Ítem-Ítem (`Item-Item.py`) + Usuario-Usuario (`user_user.py`)
> **Servidor unificado:** `main.py` (sirve ambos modelos en el mismo puerto)

---

## Requisitos previos

### Python (backend)
```bash
pip install fastapi uvicorn pandas numpy scikit-learn scikit-surprise
```

### Node.js (frontend)
```bash
# Solo la primera vez (instala dependencias):
cd frontend
npm install
```

### Dataset
Los archivos CSV del dataset MovieLens 20M deben estar en `backend/dat/`:
```
backend/dat/
  rating.csv    (20M filas)
  movie.csv     (27k filas)
```

---

## PASO 1 — Construir el caché Ítem-Ítem

> **Solo necesario la primera vez** o cuando quieras reconstruir el modelo desde cero.
> Tiempo estimado: **15–30 minutos**

```bash
cd backend
python item_item.py build
```

**Qué hace:**
1. Carga `rating.csv` y `movie.csv`
2. Muestreo estratificado del 30% de usuarios (semilla 505)
3. Split train/test 70/30
4. Filtra películas con ≥ 20 ratings → 8,798 películas
5. Construye matriz usuario-ítem (41,562 × 8,798)
6. Calcula las 3 matrices de similitud: coseno, Pearson y Jaccard
7. Calcula pesos McLaughlin (γ=25) precalculados
8. Serializa todo en `backend/item_item_cache.pkl` (~2–3 GB)

**Salida esperada al terminar:**
```
================================================================
RESUMEN DEL MODELO
  Usuarios : 41,562
  Películas: 8,798
  Densidad : 1.13%

  RMSE Coseno:  0.XXXX | MAE Coseno:  0.XXXX
  RMSE Pearson: 0.XXXX | MAE Pearson: 0.XXXX
  RMSE Jaccard: 0.XXXX | MAE Jaccard: 0.XXXX
================================================================
```

---

## PASO 2 — Construir el caché Usuario-Usuario

> **Solo necesario la primera vez** o cuando quieras reconstruir el modelo desde cero.
> Tiempo estimado: **~45 minutos**

```bash
cd backend
python user_user.py build
```

**Qué hace:**
1. Carga `rating.csv` y `movie.csv`
2. Muestreo estratificado del 30% de usuarios (misma semilla 505 → mismos usuarios que Ítem-Ítem)
3. Filtra a los **2000 usuarios más activos** (necesario para que la matriz de similitud sea manejable)
4. Entrena modelo `KNNWithMeans` con `pearson_baseline`, k=50, γ=200 (McLaughlin)
5. Precalcula los top-50 vecinos de cada usuario (para el modal de explicación)
6. Serializa en `backend/model_cache.pkl` (~130 MB)

**Salida esperada al terminar:**
```
========================================================
model_cache.pkl listo
  Usuarios en modelo   : 2,000
  Usuarios en historial: 41,562
  Películas en catálogo: 27,278
  Modelo: pearson_baseline k=50 gamma=200
========================================================
```

---

## PASO 3 — Iniciar el backend

> Requiere que ambos cachés estén construidos (`item_item_cache.pkl` y `model_cache.pkl`).

```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

**Al arrancar verás (~30–60 segundos de carga):**
```
Cargando model_cache.pkl (User-User)...
  User-User OK — 41,562 usuarios
Cargando item_item_cache.pkl (Item-Item)...
  Item-Item OK — 41,562 usuarios | matriz (41562, 8798)
Servidor listo. Usuarios: 41,562
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**El servidor queda disponible en:** `http://localhost:8000`

**Documentación interactiva (Swagger):** `http://localhost:8000/docs`

---

## PASO 4 — Iniciar el frontend

> En una **nueva terminal**, con el backend ya corriendo.

```bash
cd frontend
npm install (la primera vez)
npm start
```

Abre automáticamente `http://localhost:3000` en el navegador.

---

## Resumen de comandos

```
# Terminal 1 — Backend
cd backend
python Item-Item.py build        # solo la primera vez (~15-30 min)
python user_user.py build        # solo la primera vez (~45 min)
python -m uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install                      # solo la primera vez
npm start
```

---

## Verificar que todo funciona

```bash
# El backend responde
curl http://localhost:8000/api/health

# Buscar una película
curl "http://localhost:8000/api/movies/search?q=inception"

# Login de usuario
curl -X POST http://localhost:8000/api/login \
  -H "Content-Type: application/json" \
  -d '{"userId": 1}'

# Recomendaciones Ítem-Ítem (Pearson, k=20)
curl "http://localhost:8000/api/users/1/recommendations?model=item-item&similarity=pearson&k=20&limit=5"

# Recomendaciones Usuario-Usuario
curl "http://localhost:8000/api/users/1/recommendations?model=user-user&limit=5"

# Explicación de una recomendación
curl "http://localhost:8000/api/users/1/recommendations/318/explain?similarity=pearson&k=20"
```

---

## Archivos generados

| Archivo | Cuándo se crea | Tamaño | Descripción |
|---------|---------------|--------|-------------|
| `backend/item_item_cache.pkl` | `python Item-Item.py build` | ~2–3 GB | Caché del modelo Ítem-Ítem |
| `backend/model_cache.pkl` | `python user_user.py build` | ~130 MB | Caché del modelo Usuario-Usuario |
| `backend/user_ratings_extra.json` | Al guardar ratings en la app | Variable | Ratings nuevos persistidos |
| `backend/nuevos_usuarios.json` | Al crear usuarios en la app | Variable | Usuarios creados vía API |

---

## Documentación de los modelos

- **Ítem-Ítem:** [`scripts/documentacion_item_item.md`](scripts/documentacion_item_item.md)
- **Usuario-Usuario:** [`scripts/documentacion_user_user.md`](scripts/documentacion_user_user.md)

---

## Comportamientos esperados del sistema

### Actualizar recomendaciones después de ratear una película

Al calificar una película y pulsar **"Actualizar Recomendaciones"**, el sistema puede tardar **hasta 5 minutos** en responder. Esto es normal y se debe a:

- **Ítem-Ítem:** Recalcula las 2 fases del algoritmo (candidatos vectorizados + 100 predicciones) sobre DataFrames grandes en memoria. Con muchos ratings acumulados, la operación matricial puede tardar varios segundos.
- **Usuario-Usuario:** Re-entrena el modelo Surprise completo desde cero (re-calcula la matriz de similitud 2,000 × 2,000 entera). No existe actualización incremental — es un reentrenamiento total.

No cerrar la aplicación ni el servidor mientras esperas la respuesta.

---

### Usuario-Usuario no muestra recomendaciones hasta el primer rating

Si entras con un usuario nuevo (creado desde la app) y usas el modelo **Usuario-Usuario**, es posible que no aparezca ninguna recomendación hasta que califiques al menos una película. Esto ocurre porque:

- El modelo User-User está entrenado únicamente con los **2,000 usuarios más activos** del dataset.
- Un usuario recién creado no está en ese conjunto de entrenamiento.
- Hasta que el usuario no tiene ratings registrados en el trainset de Surprise, el modelo devuelve `was_impossible=True` para todas las películas y la lista queda vacía.
- Al calificar la primera película y pulsar **"↻ Actualizar"**, el modelo se re-entrena incluyendo al nuevo usuario y las recomendaciones aparecen.

> Para usuarios nuevos se recomienda usar el modelo **Ítem-Ítem**, que funciona desde el primer rating sin necesidad de reentrenamiento.

---

### Usuarios que funcionan bien en cada modelo desde el inicio

**Ítem-Ítem** funciona correctamente para **cualquier usuario** desde el primer momento, sin importar el ID ni la cantidad de ratings previos.

**Usuario-Usuario** solo funciona bien desde el inicio para los **2,000 usuarios más activos** que forman la matriz del modelo. De ese grupo, los primeros 20 (útiles para pruebas) son:

```
134, 156, 271, 294, 348, 394, 587, 631, 637, 648,
710, 768, 847, 892, 982, 1154, 1185, 1200, 1296, 1421
```

Para cualquier otro ID (incluyendo usuarios creados desde la app), el modelo Usuario-Usuario necesita que el usuario califique al menos una película y pulse **"↻ Actualizar"** antes de mostrar recomendaciones.

---

## Solución de problemas frecuentes

| Problema | Causa | Solución |
|----------|-------|----------|
| `ModuleNotFoundError: surprise` | Surprise no instalado | `pip install scikit-surprise` |
| `FileNotFoundError: item_item_cache.pkl` | No se construyó el caché | Ejecutar `python Item-Item.py build` |
| `FileNotFoundError: model_cache.pkl` | No se construyó el caché | Ejecutar `python user_user.py build` |
| Backend tarda mucho en arrancar | Caché grande (~3 GB) cargando | Esperar 60–90 segundos |
| `npm start` falla | `node_modules` ausentes | Ejecutar `npm install` primero |
| `MemoryError` al construir | RAM insuficiente | Se necesitan al menos 8 GB libres |
| Recomendaciones User-User lentas | Normal (~3-5 seg/petición) | Es por diseño del algoritmo |
