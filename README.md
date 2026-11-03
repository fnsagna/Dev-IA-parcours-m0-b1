# Escale Atlantique — API de prédiction d'annulation

> Squelette de départ du brief **M0-B1** (FastIA). 

## Le contexte

Escale Atlantique (un hôtel urbain et un resort) subit environ une annulation pour trois réservations. Son équipe data a entraîné un modèle qui estime la probabilité d'annulation d'une réservation. Ce repo expose ce modèle via une API REST.

## Structure

```
app/
  main.py          # application FastAPI (lifespan, /health, /predict à compléter)
  schemas.py       # schémas Pydantic (entrées à contraindre)
model/
  cancellation_model.joblib   # modèle livré (pipeline scikit-learn) — ne pas modifier
  model_info.json             # fiche d'identité du modèle
  train_baseline.py           # script d'entraînement, pour information
data/
  hotel_bookings_sample.csv   # extrait du jeu d'entraînement (1 000 réservations)
  exemple_reservation.json    # exemple de corps de requête pour /predict
tests/
  conftest.py      # fixture `client` (TestClient)
  test_health.py   # 2 tests fournis
Dockerfile         # à compléter
pytest.ini         # configuration de pytest
requirements.txt   # dépendances (versions figées)
```

## Démarrage rapide

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    — macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- Santé du service : http://localhost:8000/health
- Documentation interactive : http://localhost:8000/docs

```bash
pytest -v
```

## Lancement avec Docker

```bash
docker build -t fastia-annulation:dev .
docker run --rm -p 8000:8000 fastia-annulation:dev
```
#### Exemple d'appel à /predict : Vérifier depuis votre machine (hors conteneur)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/exemple_reservation.json
``` 
####  Réponse attendue :

{"annulation": false, "probabilite_annulation": 0.12}


vérifiez d'abord que `data/exemple_reservation.json` correspond bien au schéma `Booking` — sinon adaptez l'exemple


## Données

Extrait de *Hotel Booking Demand* — N. Antonio, A. de Almeida, L. Nunes (2019), *Data in Brief* 22, 41-49. Licence CC BY 4.0.
