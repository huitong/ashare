CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    day TEXT PRIMARY KEY, complete INTEGER NOT NULL DEFAULT 0 CHECK (complete IN (0,1))
);
CREATE TABLE IF NOT EXISTS securities (
    code TEXT PRIMARY KEY, name TEXT NOT NULL, listed TEXT NOT NULL, delisted TEXT
);
CREATE TABLE IF NOT EXISTS bars (
    day TEXT NOT NULL REFERENCES sessions(day), code TEXT NOT NULL REFERENCES securities(code),
    open REAL NOT NULL CHECK(open > 0), high REAL NOT NULL, low REAL NOT NULL,
    close REAL NOT NULL CHECK(close > 0), prev_close REAL,
    volume REAL NOT NULL CHECK(volume >= 0), amount REAL NOT NULL CHECK(amount >= 0),
    market_cap REAL NOT NULL CHECK(market_cap > 0), adj REAL NOT NULL CHECK(adj > 0),
    limit_up REAL, limit_down REAL, limit_known INTEGER NOT NULL CHECK(limit_known IN (0,1)),
    PRIMARY KEY(day, code)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS bars_code_day ON bars(code, day);

CREATE TABLE IF NOT EXISTS industry_membership (
    code TEXT NOT NULL REFERENCES securities(code), industry TEXT NOT NULL,
    effective_from TEXT NOT NULL, effective_to TEXT, known_at TEXT NOT NULL,
    PRIMARY KEY(code, industry, effective_from)
);
CREATE INDEX IF NOT EXISTS membership_from ON industry_membership(effective_from);

CREATE TABLE IF NOT EXISTS hotspots (
    event_id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('industry','code')),
    target TEXT NOT NULL, published_at TEXT NOT NULL, available_at TEXT NOT NULL,
    strength REAL NOT NULL CHECK(strength > 0), title TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS hotspots_available ON hotspots(available_at);

CREATE TABLE IF NOT EXISTS holding_snapshots (
    asof TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS holdings (
    asof TEXT NOT NULL REFERENCES holding_snapshots(asof) ON DELETE CASCADE,
    code TEXT NOT NULL REFERENCES securities(code), shares INTEGER NOT NULL CHECK(shares > 0),
    PRIMARY KEY(asof, code)
);
