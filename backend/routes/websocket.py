from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services import prediction_service


router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/predict")
async def prediction_socket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            power = float(payload.get("power", 0))
            time = float(payload.get("time", 0))
            antenna_type = str(payload.get("antenna_type", "Other"))
            result = prediction_service.predict_compare(power=power, time=time, antenna_type=antenna_type)
            for item in result["results"]:
                await websocket.send_json({"type": "partial", "result": item})
            await websocket.send_json({"type": "final", "ensemble": result["ensemble_prediction"]})
    except WebSocketDisconnect:
        return
