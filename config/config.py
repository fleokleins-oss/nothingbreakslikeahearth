"""
config/config.py
APEX PREDATOR NEO v3 – Configuração centralizada via Pydantic Settings.
Toda variável de ambiente é validada no boot; se faltar algo crítico, o sistema
não sobe e exibe mensagem clara.
"""
from __future__ import annotations
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class ApexConfig(BaseSettings):
    # ── Modo ─────────────────────────────────────────────
    TESTNET: bool = True

    # ── Credenciais ──────────────────────────────────────
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_TESTNET_API_KEY: str = ""
    BINANCE_TESTNET_API_SECRET: str = ""

    # ── Redis ────────────────────────────────────────────
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    # ── Capital / Risco ──────────────────────────────────
    CAPITAL_TOTAL: float = 22.00
    MAX_POR_CICLO: float = 8.00
    MAX_DRAWDOWN_PCT: float = 4.0
    ROBIN_HOOD_COOLDOWN_S: int = 1800

    # ── Scanner ──────────────────────────────────────────
    SCAN_INTERVAL_MS: int = 45
    MIN_PROFIT_PCT: float = 0.08
    MIN_CONFLUENCE_SCORE: float = 65.0

    # ── Taxas ────────────────────────────────────────────
    MAKER_FEE: float = 0.00075
    TAKER_FEE: float = 0.00075

    # ── Auto-Earn ────────────────────────────────────────
    AUTO_EARN_MIN_PROFIT: float = 0.10

    # ── Logging ──────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_ROTATION: str = "50 MB"
    LOG_RETENTION: str = "7 days"

    # ── Docker ENV ───────────────────────────────────────
    APEX_ROLE: str = "scanner"
    APEX_REGION: str = "curitiba"

    # ── Ativos para descoberta de triângulos ─────────────
    BASE_ASSETS: List[str] = [
        "BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA",
        "AVAX", "DOT", "MATIC", "LINK", "SHIB", "TRX",
        "WIF", "BONK", "PEPE", "NEAR", "APT", "SUI",
        "ARB", "OP", "FIL", "ATOM", "UNI", "LTC",
    ]
    QUOTE_ASSETS: List[str] = ["USDT", "BRL", "BTC", "ETH", "BNB", "FDUSD"]

    # ── Canais Redis ─────────────────────────────────────
    CH_OPPORTUNITIES: str = "apex:v3:opportunities"
    CH_EXECUTIONS: str = "apex:v3:executions"
    CH_HEARTBEAT: str = "apex:v3:heartbeat"
    CH_RISK: str = "apex:v3:risk"
    CH_EARN: str = "apex:v3:earn"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    # ── Propriedades derivadas ───────────────────────────
    @property
    def api_key(self) -> str:
        return self.BINANCE_TESTNET_API_KEY if self.TESTNET else self.BINANCE_API_KEY

    @property
    def api_secret(self) -> str:
        return self.BINANCE_TESTNET_API_SECRET if self.TESTNET else self.BINANCE_API_SECRET

    @property
    def fee_per_leg(self) -> float:
        """Taxa média por perna (maker+taker)/2."""
        return (self.MAKER_FEE + self.TAKER_FEE) / 2

    @property
    def fee_3_legs(self) -> float:
        """Taxa total estimada para 3 pernas."""
        return self.fee_per_leg * 3


# Singleton importável em qualquer lugar
cfg = ApexConfig()
