// GET /api/history: the 20 most recent predictions, read from D1
export async function onRequestGet({ env }) {
  const { results } = await env.DB.prepare(
    "SELECT id, created_at, file_name, score, threshold, is_anomaly FROM predictions ORDER BY created_at DESC LIMIT 20",
  ).all();
  return Response.json(results);
}
