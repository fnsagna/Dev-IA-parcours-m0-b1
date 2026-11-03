import pytest

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}

def test_openapi_documentation_is_available(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]

@pytest.fixture
def reservation_valide():
    return {
        "hotel": "Resort Hotel",
        "lead_time": 107,
        "arrival_date_month": "July",
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "market_segment": "Online TA",
        "deposit_type": "No Deposit",
        "customer_type": "Transient",
        "country": "PRT",
        "adr": 110.7,
        "previous_cancellations": 0,
        "total_of_special_requests": 2,
        "required_car_parking_spaces": 0,
    }

def test_predict_reservation_valide(client, reservation_valide):
    response = client.post("/predict", json=reservation_valide)
    assert response.status_code == 200
    data = response.json()
    assert "annulation" in data
    assert 0 <= data["probabilite_annulation"] <= 1

def test_predict_hotel_invalide(client, reservation_valide):
    reservation_valide["hotel"] = "Hotel Inconnu"
    response = client.post("/predict", json=reservation_valide)
    assert response.status_code == 422

@pytest.mark.parametrize("hotel", ["City Hotel", "Resort Hotel"])
def test_predict_les_deux_hotels(client, reservation_valide, hotel):
    reservation_valide["hotel"] = hotel
    response = client.post("/predict", json=reservation_valide)
    assert response.status_code == 200

