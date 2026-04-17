from pydantic import ValidationError

from schemas.prediction import PredictionRequest


def test_prediction_schema_rejects_zero_power():
    try:
        PredictionRequest(power=0, time=5, antenna_type="Other")
        assert False
    except ValidationError:
        assert True
