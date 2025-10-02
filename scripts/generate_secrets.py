#!/usr/bin/env python3
"""
Generate cryptographically secure secrets for production deployment
Run this script to generate all required secret keys and tokens
"""

import secrets
import base64
import hashlib
import argparse
import json
from pathlib import Path

def generate_secret_key(length: int = 32) -> str:
    """Generate a cryptographically secure random key"""
    return secrets.token_urlsafe(length)

def generate_hex_key(length: int = 32) -> str:
    """Generate a hex secret key"""
    return secrets.token_hex(length)

def generate_api_token() -> str:
    """Generate API token"""
    return secrets.token_urlsafe(32)

def generate_jwt_secret() -> str:
    """Generate JWT secret"""
    return base64.b64encode(secrets.token_bytes(32)).decode('utf-8')

def generate_password(length: int = 16) -> str:
    """Generate a secure password"""
    import string
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def main():
    parser = argparse.ArgumentParser(description='Generate secure secrets for production')
    parser.add_argument('--format', choices=['env', 'json'], default='env',
                       help='Output format (env variables or JSON)')
    parser.add_argument('--output', type=str,
                       help='Output file path')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output')

    args = parser.parse_args()

    # Generate all secrets
    secrets_data = {
        'SECRET_KEY': generate_secret_key(32),
        'JWT_SECRET_KEY': generate_jwt_secret(),
        'API_TOKEN': generate_api_token(),
        'AES_KEY': generate_hex_key(32),
        'SESSION_SECRET': generate_secret_key(32),
        'ENCRYPTION_KEY': generate_hex_key(32),
        'WEBHOOK_SECRET': generate_secret_key(32),
        'ADMIN_PASSWORD': generate_password(20)
    }

    # Output based on format
    if args.format == 'json':
        output_content = json.dumps(secrets_data, indent=2)
        if not args.quiet:
            print("=== GENERATED SECRETS (JSON) ===")
            print(output_content)
    else:
        output_content = "# ==================================\n"
        output_content += "# GENERATED SECRETS - DO NOT COMMIT\n"
        output_content += "# ==================================\n\n"

        for key, value in secrets_data.items():
            output_content += f"{key}={value}\n"

        output_content += "\n# =============================\n"
        output_content += "# USAGE INSTRUCTIONS\n"
        output_content += "# =============================\n"
        output_content += "# 1. Copy these values to your .env files\n"
        output_content += "# 2. Store securely in your deployment platform\n"
        output_content += "# 3. Rotate keys every 90 days for security\n"
        output_content += "# 4. Use different keys for staging/production\n"

        if not args.quiet:
            print("=== GENERATED SECRETS (ENV FORMAT) ===")
            print(output_content)

    # Write to file if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(output_content)

        if not args.quiet:
            print(f"✅ Secrets written to: {output_path}")

    # Security warnings
    if not args.quiet:
        print("\n" + "="*50)
        print("SECURITY WARNINGS:")
        print("- NEVER commit these secrets to version control")
        print("- Store secrets securely (Railway/Vercel env vars)")
        print("- Rotate keys regularly (every 90 days)")
        print("- Use different secrets for each environment")
        print("- Monitor for secret leaks in logs")
        print("="*50)

    return secrets_data

if __name__ == "__main__":
    main()
