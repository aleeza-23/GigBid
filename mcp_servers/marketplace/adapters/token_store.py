"""Encrypted, per-user OAuth token storage.

A4 requires OAuth tokens stored encrypted per user; E4 requires that they are never
logged. Both are easy to get wrong in the direction that only shows up in a breach, so the
rules are enforced here rather than trusted to callers:

* Tokens are encrypted at rest with Fernet (AES-128-CBC with an HMAC), keyed from
  TOKEN_ENCRYPTION_KEY. No key, no storage: a missing key raises instead of silently
  falling back to plaintext, because a silent fallback is how plaintext tokens reach a
  disk nobody audits.
* Rows are scoped by user id, and every read takes a user id. There is no "get all
  tokens" call, so a tenant-scoping bug has to be written on purpose.
* __repr__ never includes the token, so a token cannot reach a log through an f-string or
  an exception rendering a dataclass.

Generate a key with:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(
    os.getenv(
        "OAUTH_TOKEN_DB",
        Path(__file__).resolve().parents[3] / "data" / "audit" / "oauth_tokens.sqlite3",
    )
)


class TokenStoreError(RuntimeError):
    """Misconfiguration, a missing token, or a key that cannot decrypt a row."""


@dataclass
class OAuthToken:
    """One user's credentials for one provider."""

    user_id: str
    provider: str
    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expires_at: float | None = None
    scope: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - defensive, but the point of the class
        return (
            f"OAuthToken(user_id={self.user_id!r}, provider={self.provider!r}, "
            f"expires_in={self.seconds_remaining():.0f}s, token=[redacted])"
        )

    __str__ = __repr__

    def seconds_remaining(self) -> float:
        if self.expires_at is None:
            return float("inf")
        return max(0.0, self.expires_at - time.time())

    def is_expired(self, skew: float = 60.0) -> bool:
        """Treat a token as expired a minute early, so a refresh beats a 401."""
        return self.seconds_remaining() <= skew


def _fernet():
    """Build the cipher, or explain exactly what is missing."""
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise TokenStoreError(
            "live mode needs the 'cryptography' package: pip install -e '.[live]'"
        ) from exc

    raw = os.getenv("TOKEN_ENCRYPTION_KEY", "").strip()
    if not raw:
        raise TokenStoreError(
            "TOKEN_ENCRYPTION_KEY is not set. Generate one with:\n"
            '  python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())"\n'
            "Refusing to store OAuth tokens unencrypted."
        )
    try:
        return Fernet(raw.encode())
    except Exception:
        # Accept a human-chosen passphrase by deriving a proper key from it, rather than
        # making the reviewer generate base64 by hand to start the app.
        derived = base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest())
        return Fernet(derived)


class TokenStore:
    """SQLite-backed, encrypted, scoped by user."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or DEFAULT_PATH)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS oauth_tokens (
                user_id    TEXT NOT NULL,
                provider   TEXT NOT NULL,
                payload    BLOB NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (user_id, provider)
            )
            """
        )
        self._conn.commit()

    def save(self, token: OAuthToken) -> None:
        blob = _fernet().encrypt(
            json.dumps(
                {
                    "access_token": token.access_token,
                    "refresh_token": token.refresh_token,
                    "expires_at": token.expires_at,
                    "scope": token.scope,
                }
            ).encode()
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO oauth_tokens (user_id, provider, payload, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(user_id, provider) "
                "DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (token.user_id, token.provider, blob, time.time()),
            )
            self._conn.commit()

    def load(self, user_id: str, provider: str = "upwork") -> OAuthToken | None:
        """This user's token, or None. There is deliberately no all-users variant."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT payload FROM oauth_tokens WHERE user_id=? AND provider=?",
                (user_id, provider),
            )
            row = cur.fetchone()
        if row is None:
            return None
        try:
            data: dict[str, Any] = json.loads(_fernet().decrypt(row[0]).decode())
        except Exception as exc:
            raise TokenStoreError(
                f"cannot decrypt the stored {provider} token for {user_id}: "
                "TOKEN_ENCRYPTION_KEY has probably changed. The user must reconnect."
            ) from exc
        return OAuthToken(user_id=user_id, provider=provider, **data)

    def delete(self, user_id: str, provider: str = "upwork") -> None:
        """Disconnect an account. The settings screen calls this."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM oauth_tokens WHERE user_id=? AND provider=?", (user_id, provider)
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
