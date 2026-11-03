"""Schémas Pydantic : format des entrées et des sorties de l'API."""

from typing import Literal

from pydantic import BaseModel, Field

class Booking(BaseModel):
    """Une réservation, telle que saisie dans le logiciel de gestion hôtelière.

    Les noms de champs sont ceux du jeu de données d'entraînement :
    le modèle attend exactement ces colonnes.

    TODO (tâche 3) : ces champs ne sont pas contraints. Une valeur absurde
    (type d'hôtel inconnu, nombre d'adultes négatif…) passe la validation
    et arrive jusqu'au modèle. Ajoute les contraintes nécessaires
    (indices : `Literal`, `Field(ge=..., le=...)`).
    """
    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(ge=0)
    arrival_date_month: Literal[
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",]
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: int = Field(ge=0)
    market_segment: Literal[
    "Aviation", "Complementary", "Corporate", "Direct",
    "Groups", "Offline TA/TO", "Online TA",]
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    customer_type: Literal["Contract", "Group", "Transient", "Transient-Party"]
    country: str = Field(min_length=2, max_length=3)
    adr: float = Field(ge=0)
    previous_cancellations: int = Field(ge=0)
    total_of_special_requests: int = Field(ge=0)
    required_car_parking_spaces: int = Field(ge=0)


class PredictionResponse(BaseModel):
    """Réponse de /predict."""

    annulation: bool
    probabilite_annulation: float
