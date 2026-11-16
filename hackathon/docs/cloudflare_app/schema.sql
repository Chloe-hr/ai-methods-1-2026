-- One row per prediction. Run with: npx wrangler d1 execute hackathon-example-db --local --file=schema.sql
CREATE TABLE IF NOT EXISTS predictions (
  id          TEXT PRIMARY KEY,               -- a random id (crypto.randomUUID)
  created_at  TEXT NOT NULL,                  -- ISO date and time, UTC
  image_key   TEXT NOT NULL,                  -- where the uploaded image is stored in R2
  file_name   TEXT,                           -- the name of the uploaded file
  score       REAL NOT NULL,                  -- reconstruction error returned by the model server
  threshold   REAL NOT NULL,                  -- threshold used by the model server
  is_anomaly  INTEGER NOT NULL                -- 1 = anomaly, 0 = normal (SQLite has no boolean)
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
