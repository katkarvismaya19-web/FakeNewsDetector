"""Minimal user store: SQLite + salted PBKDF2 password hashes."""
import hashlib, os, re, sqlite3

DB = os.getenv("USER_DB", "users.db")


def _conn():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT, salt BLOB, pw BLOB)")
    return c


def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)


def register(name, email, pw):
    email = email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return False, "Enter a valid email address."
    if len(pw) < 8:
        return False, "Password must be at least 8 characters."
    salt = os.urandom(16)
    with _conn() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            return False, "An account with this email already exists."
        c.execute("INSERT INTO users VALUES (?,?,?,?)", (email, name.strip(), salt, _hash(pw, salt)))
    return True, "Account created. Log in to continue."


def login(email, pw):
    with _conn() as c:
        row = c.execute("SELECT name, salt, pw FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
    if row and _hash(pw, row[1]) == row[2]:
        return row[0]
    return None
