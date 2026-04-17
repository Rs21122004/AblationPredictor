export function createPredictionSocket(onMessage) {
  const ws = new WebSocket('ws://localhost:8000/ws/predict');
  ws.onmessage = (event) => {
    const parsed = JSON.parse(event.data);
    onMessage(parsed);
  };
  return ws;
}
