import os

# ── App ───────────────────────────────────────
APP_TITLE = "AI Sales Forecasting System"
APP_ICON = "📊"
APP_VERSION = "2.0.0"

# ── Paths ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SAMPLE_CSV_PATH = os.path.join(DATA_RAW_DIR, "sample_sales.csv")

# ── Forecasting defaults ──────────────────────
DEFAULT_FORECAST_PERIODS = 12
DEFAULT_MODEL = "holts"
MAX_FORECAST_PERIODS = 24
TRAIN_TEST_SPLIT_RATIO = 0.2
DEFAULT_CI_Z = 1.96  # 95% confidence interval

# ── Model hyperparameter defaults ────────────
MOVING_AVG_WINDOW = 3
EXP_SMOOTHING_ALPHA = 0.3
HOLTS_ALPHA = 0.3
HOLTS_BETA = 0.1

# ── ML Model defaults ────────────────────────
MIN_POINTS_FOR_ML_MODEL = 6
RANDOM_FOREST_N_ESTIMATORS = 300
RANDOM_FOREST_MAX_DEPTH = 8
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 2
RANDOM_FOREST_MIN_SAMPLES_LEAF = 1
RANDOM_FOREST_RANDOM_STATE = 42

# ── Best model selection ─────────────────────
BEST_MODEL_METRIC_KEY = "MAPE (%)"
BEST_MODEL_LOWER_IS_BETTER = True

# ── Anomaly detection ────────────────────────
DEFAULT_ANOMALY_METHOD = "rolling_deviation"
DEFAULT_ANOMALY_THRESHOLD = 2.5
DEFAULT_ANOMALY_ROLLING_WINDOW = 6
DEFAULT_OUTLIER_METHOD = "iqr"

# ── Forecast confidence cutoffs ──────────────
CONFIDENCE_HIGH_MIN = 75
CONFIDENCE_MODERATE_MIN = 50
CONFIDENCE_LOW_MIN = 0

# Scoring thresholds
CONFIDENCE_MAPE_EXCELLENT = 10
CONFIDENCE_MAPE_GOOD = 20
CONFIDENCE_MAPE_ACCEPTABLE = 35

CONFIDENCE_VOLATILITY_LOW = 10
CONFIDENCE_VOLATILITY_MODERATE = 20
CONFIDENCE_VOLATILITY_HIGH = 35

CONFIDENCE_MIN_DATA_STRONG = 24
CONFIDENCE_MIN_DATA_OK = 12
CONFIDENCE_MIN_DATA_WEAK = 6

CONFIDENCE_FEW_ANOMALIES = 2
CONFIDENCE_MODERATE_ANOMALIES = 5

# ── Chatbot / Gemini ─────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_TOKENS = 1024
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
CHAT_MAX_HISTORY = 10

# ── Chatbot intent keywords ──────────────────
CHATBOT_INTENT_KEYWORDS = {
    "summary": [
        "summary", "overview", "dataset", "overall", "tell me about my data"
    ],
    "trend": [
        "trend", "growth", "decline", "upward", "downward", "momentum"
    ],
    "best_category": [
        "best category", "top category", "highest category", "category performs best"
    ],
    "best_region": [
        "best region", "top region", "highest region", "region performs best"
    ],
    "comparison": [
        "compare", "comparison", "vs", "versus", "last 3 months", "last month vs"
    ],
    "best_month": [
        "best month", "highest month", "top month", "peak month"
    ],
    "anomaly": [
        "anomaly", "spike", "drop", "unusual", "outlier"
    ],
    "forecast": [
        "forecast", "predict", "next month", "next 6", "future sales"
    ],
    "scenario": [
        "scenario", "what if", "simulate", "marketing", "churn", "discount"
    ],
}

# ── Data validation thresholds ───────────────
MIN_ROWS_REQUIRED = 30
MAX_MISSING_PCT = 20.0
MAX_DUPLICATE_PCT = 5.0

# ── Scenario simulation presets ──────────────
SCENARIO_BEST = dict(
    growth=20,
    seasonality=10,
    discount=0,
    marketing=25,
    churn=0,
    new_channel=15
)

SCENARIO_BASE = dict(
    growth=8,
    seasonality=0,
    discount=-5,
    marketing=10,
    churn=3,
    new_channel=0
)

SCENARIO_WORST = dict(
    growth=-5,
    seasonality=-10,
    discount=-15,
    marketing=0,
    churn=15,
    new_channel=0
)

# ── UI colours ───────────────────────────────
COLOR_BLUE = "#38bdf8"
COLOR_GREEN = "#34d399"
COLOR_AMBER = "#fbbf24"
COLOR_ROSE = "#f87171"
COLOR_PURPLE = "#a78bfa"
COLOR_ORANGE = "#fb923c"
COLOR_INDIGO = "#818cf8"

BG_DARK = "#0a0c10"
BG_CARD = "#111827"
BORDER_COLOR = "#1f2937"