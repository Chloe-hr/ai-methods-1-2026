// POST /api/predict with a form field "image" (a file):
// 1. send it to the model server, 2. store the image in R2, 3. store the result in D1, 4. answer the page.

export async function onRequestPost({ request, env }) {
  const form = await request.formData();
  const file = form.get("image");
  if (!file || typeof file === "string") {
    return Response.json({ error: "Send an image file in the field 'image'." }, { status: 400 });
  }
  if (file.size > 5 * 1024 * 1024) {
    return Response.json({ error: "Image too large (max 5 MB)." }, { status: 413 });
  }
  const bytes = new Uint8Array(await file.arrayBuffer());

  // 1. Ask the model
  let result;
  try {
    result = await callModel(env, toBase64(bytes));
  } catch (err) {
    // The most common reason: the free model server was asleep and is still starting
    return Response.json(
      { error: "The model did not answer. If the model server was asleep it is waking up: try again in a minute.", detail: String(err) },
      { status: 503 },
    );
  }

  // 2. Store the image in R2 under a random name
  const id = crypto.randomUUID();
  const imageKey = `uploads/${id}`;
  await env.BUCKET.put(imageKey, bytes, { httpMetadata: { contentType: file.type || "application/octet-stream" } });

  // 3. Store one row in D1. The "?" placeholders keep user input out of the SQL text
  const createdAt = new Date().toISOString();
  await env.DB.prepare(
    "INSERT INTO predictions (id, created_at, image_key, file_name, score, threshold, is_anomaly) VALUES (?, ?, ?, ?, ?, ?, ?)",
  )
    .bind(id, createdAt, imageKey, file.name || null, result.score, result.threshold, result.is_anomaly ? 1 : 0)
    .run();

  // 4. Answer the page
  return Response.json({ id, created_at: createdAt, ...result });
}

// Gradio's HTTP API has two steps: start a job, then read its result
async function callModel(env, imageBase64) {
  const base = `${env.MODEL_URL}/gradio_api/call/${env.MODEL_API_NAME}`;
  const started = await fetchWithTimeout(base, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: [imageBase64] }),
  });
  if (!started.ok) throw new Error(`Model server answered ${started.status}`);
  const { event_id } = await started.json();

  const stream = await fetchWithTimeout(`${base}/${event_id}`);
  const text = await stream.text(); // "event: complete\ndata: [ {...} ]"
  const complete = text.split("event: complete")[1];
  if (!complete) throw new Error(`No result from the model server: ${text.slice(0, 200)}`);
  const data = JSON.parse(complete.split("data:")[1].trim());
  return data[0];
}

async function fetchWithTimeout(url, options = {}, ms = 60000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function toBase64(bytes) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}
