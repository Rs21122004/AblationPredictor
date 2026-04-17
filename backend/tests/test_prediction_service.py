from services import prediction_service


def test_batch_prediction_handles_invalid_row():
    result = prediction_service.predict_batch([{"power": -1, "time": 3}])
    assert result["failed"] == 1
