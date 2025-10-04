## Security Policy

### Handling Secrets
- Never commit secrets to the repository.
- Store all credentials as environment variables in deployment platforms (Vercel, Railway) or in GitHub Actions secrets.
- Local development: use `.env.local` (gitignored).

### Required Secrets
- OPENAI_API_KEY, GEMINI_API_KEY / GOOGLE_API_KEY
- SECRET_KEY, JWT_SECRET_KEY, API_TOKEN
- DATABASE_URL, NEON_DATABASE_URL
- BLOB_READ_WRITE_TOKEN, SMTP_PASS
- VERCEL_TOKEN, RAILWAY_TOKEN (CI/CD only)

### Rotation Procedure
1. Generate new secrets using `python scripts/generate_secrets.py` (or provider dashboards).
2. Update environment variables in Vercel/Railway and GitHub Secrets.
3. Invalidate/revoke old secrets in respective dashboards.
4. Re-deploy and verify.

### Scanning & Prevention
- GitHub Push Protection is enabled to block secret pushes.
- CI runs Gitleaks and TruffleHog (`.gitleaks.toml` allowlists only placeholders/examples).
- Optionally install a local pre-commit hook to run `gitleaks detect`.

### Reporting a Vulnerability
Please open a private security advisory or contact the maintainers directly. Do not open public issues for sensitive disclosures.

