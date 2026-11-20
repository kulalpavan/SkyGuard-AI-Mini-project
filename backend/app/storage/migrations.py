"""Versioned additive migrations. Credential erasure is intentionally irreversible.

Existing password hashes remain usable and are upgraded to Argon2id on login.
Take a database backup before upgrade; never restore plaintext credential columns.
"""


def migrate(conn):
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
    cur.execute("SELECT version FROM schema_migrations")
    versions = {r["version"] for r in cur.fetchall()}
    if 1 not in versions:
        cur.execute("UPDATE stations SET access_key = '' WHERE access_key IS NOT NULL")
        if conn.is_postgres:
            cur.execute("ALTER TABLE stations ALTER COLUMN access_key SET DEFAULT ''")
            cur.execute("ALTER TABLE stations ADD CONSTRAINT station_no_plaintext CHECK (access_key IS NULL OR access_key = '')")
        else:
            for action in ("INSERT", "UPDATE"):
                cur.execute(f"""CREATE TRIGGER IF NOT EXISTS no_plaintext_{action.lower()}
                    BEFORE {action} ON stations WHEN NEW.access_key IS NOT NULL AND NEW.access_key != ''
                    BEGIN SELECT RAISE(ABORT, 'plaintext credentials forbidden'); END""")
        cur.execute("INSERT INTO schema_migrations(version) VALUES (1)")
    if 2 not in versions:
        cur.execute("""CREATE TABLE IF NOT EXISTS observations (
            observation_id TEXT PRIMARY KEY, station_id TEXT NOT NULL REFERENCES stations(station_id),
            source_timestamp TEXT NOT NULL, received_at TEXT NOT NULL, source TEXT NOT NULL,
            payload_hash TEXT NOT NULL, raw_data TEXT NOT NULL, assessment_data TEXT NOT NULL,
            UNIQUE(station_id, source_timestamp, source))""")
        cur.execute("CREATE INDEX IF NOT EXISTS observations_station_time ON observations(station_id, source_timestamp)")
        cur.execute("""CREATE TABLE IF NOT EXISTS observation_quarantine (
            id TEXT PRIMARY KEY, station_id TEXT NOT NULL, received_at TEXT NOT NULL,
            reason TEXT NOT NULL, raw_data TEXT NOT NULL)""")
        cur.execute("CREATE TABLE IF NOT EXISTS station_pipeline_state (station_id TEXT PRIMARY KEY REFERENCES stations(station_id), source_timestamp TEXT NOT NULL, normal_streak INTEGER NOT NULL DEFAULT 0)")
        cur.execute("CREATE TABLE IF NOT EXISTS fault_ledger (id TEXT PRIMARY KEY, station_id TEXT NOT NULL, fault_type TEXT NOT NULL, source_timestamp TEXT NOT NULL, expected_class TEXT NOT NULL, observed_class TEXT NOT NULL, evidence TEXT NOT NULL)")
        cur.execute("INSERT INTO schema_migrations(version) VALUES (2)")
    if 3 not in versions:
        # Serialize per-station model jobs without preventing archived history.
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_running_training ON training_jobs(station_id) WHERE status = 'RUNNING'")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_active_model ON model_registry(station_id) WHERE status = 'ACTIVE'")
        cur.execute("INSERT INTO schema_migrations(version) VALUES (3)")
    if 4 not in versions:
        # Drift tracking (§21): extend the seed-era drift_metrics /
        # retraining_candidates tables to the full §24.3 shape. The two
        # tables were created twice in init_db with different columns
        # (SQLite branch vs extended tables), so add whatever is missing
        # rather than assuming one shape. Existing rows and the legacy
        # record_drift_metrics()/register_retraining_candidate() writers
        # keep working: the new columns are nullable with defaults.
        def _add_column(table, column, ddl):
            if conn.is_postgres:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {ddl}")
            else:
                try:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                except Exception:
                    pass  # column already exists
        for column, ddl in (
            ("model_version", "TEXT"),
            ("model_type", "TEXT"),
            ("window_days", "INTEGER"),
            ("computed_at", "TEXT"),
            ("psi_temperature", "REAL"),
            ("psi_humidity", "REAL"),
            ("psi_pressure", "REAL"),
            ("residual_mean_shift", "REAL"),
            ("residual_variance_ratio", "REAL"),
            ("anomaly_rate_reference", "REAL"),
            ("ks_statistic", "REAL"),
            ("feature_availability", "REAL"),
            ("severity", "TEXT"),
            ("triggered_candidate", "INTEGER DEFAULT 0"),
            ("metrics_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("is_partial", "INTEGER DEFAULT 0"),
            ("score_mean", "REAL"),
            ("score_std", "REAL"),
        ):
            _add_column("drift_metrics", column, ddl)
        for column, ddl in (
            ("trigger_reason", "TEXT"),
            ("priority", "TEXT DEFAULT 'LOW'"),
            ("status", "TEXT DEFAULT 'QUEUED'"),
            ("gate_status", "TEXT DEFAULT 'PENDING'"),
            ("gate_results", "TEXT NOT NULL DEFAULT '{}'"),
            ("artifact_hash", "TEXT"),
        ):
            _add_column("retraining_candidates", column, ddl)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_drift_station_time ON drift_metrics(station_id, computed_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_drift_station_model ON drift_metrics(station_id, model_type, computed_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_station_status ON retraining_candidates(station_id, status)")
        cur.execute("INSERT INTO schema_migrations(version) VALUES (4)")
