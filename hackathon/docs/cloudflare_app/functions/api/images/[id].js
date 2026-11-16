// GET /api/images/<id>: the uploaded image, read back from R2 (behind the password like everything else)
export async function onRequestGet({ params, env }) {
  const object = await env.BUCKET.get(`uploads/${params.id}`);
  if (!object) {
    return new Response("Not found", { status: 404 });
  }
  return new Response(object.body, {
    headers: { "Content-Type": object.httpMetadata?.contentType || "application/octet-stream" },
  });
}
