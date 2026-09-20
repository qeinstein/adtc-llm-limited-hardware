# Jamii Afya privacy (PRIVACY.md)

Plain-language rules; not legal advice. Nigerian data-protection (NDPR/NDPA)
obligations are tracked in documentation without claiming legal review.

## Principles

- No live user conversations used for training by default. Ever.
- No PHI/PII in committed datasets (`build_dataset.py` drops phone/email/
  address patterns; counts reported in the ledger, content never stored).
- Future data donation requires explicit opt-in with a separate consent
  record; no such pipeline exists yet.

## Logging

- Web/app logs contain operational telemetry only (latency, tokens, and RSS).
- Prompt and response text are not written to application logs.
- No prompt/response logging in CI artifacts. Secrets (tokens, keys) are
  never printed; env-only (`HF_TOKEN`, `KAGGLE_*`).
- Retention: operational logs 30 days; anything containing health content
  is a bug — report and purge.

## Deployment notes

- All inference is local (CPU + local SSD); no health text leaves the
  machine in the reference deployment.
- The RAG corpus and model artifacts contain no user data.
