# time-series-prediction

A Django web application for forecasting meteorological time series. Users sign in, upload a spreadsheet of daily weather and soil-moisture readings, pick a variable and a forecast horizon, and get an automatic ARIMA forecast with diagnostics and downloadable results. The interface is in Portuguese.

## Features

- Session-based authentication (login and registration)
- Upload of a spreadsheet and selection of the variable to model: temperature (average, maximum, minimum), humidity, global solar radiation, wind direction and speed, rainfall, grass temperature and evapotranspiration (ET0)
- Automatic model selection with `pmdarima` `auto_arima`, with optional seasonality (period of 12)
- Diagnostics: ADF and KPSS stationarity tests, ACF and PACF plots, autocorrelation plot and seasonal decomposition
- Forecast plots, with RMSE evaluation when validation is enabled
- Export of the results as PNG, XLSX and a ZIP archive
- Deployment scripts for Docker, Gunicorn and Nginx, plus helper scripts to check the server and free memory

## Tech stack

Django 3.2, pandas, NumPy, statsmodels, pmdarima, scikit-learn, matplotlib, seaborn, openpyxl, SQLAlchemy, PostgreSQL (through `dj-database-url`), Gunicorn, WhiteNoise. The UI is based on the Datta Able free Bootstrap admin template, via the [AppSeed Django dashboard](https://github.com/app-generator/django-datta-able).

## Getting started

Requirements: Python 3 (see `runtime.txt` for the version used in deployment).

```bash
python -m venv env
source env/bin/activate        # on Windows: .\env\Scripts\activate
pip install -r requirements.txt

python manage.py makemigrations
python manage.py migrate
python manage.py runserver     # http://127.0.0.1:8000/
```

Open the app, register a new user on the registration page and sign in to unlock the private pages. Configuration is read from environment variables or a `.env` file (via `python-decouple`).

### With Docker

```bash
docker compose up --build -d
```

The published port is defined in `docker-compose.yml`.

### Production servers

```bash
gunicorn --config gunicorn-cfg.py tsprediction.wsgi:application
```

An Nginx configuration is provided in `nginx/` and a `Procfile` for platform deployments.

## Project structure

| Path | Purpose |
|---|---|
| `tsprediction/` | Django project (settings, URLs, WSGI and ASGI) |
| `apps/app/` | Upload, ARIMA modelling, plots and exports (`views.py`) |
| `apps/authentication/` | Login and registration |
| `predictions/` | Django app for stored predictions |
| `core/` | Templates and static assets |
| `nginx/`, `scripts/` | Nginx configuration and server helper scripts |
| `Dockerfile`, `docker-compose.yml`, `gunicorn-cfg.py`, `Procfile` | Deployment files |

## License

See [LICENSE.md](LICENSE.md). The base template is provided by AppSeed and CodedThemes.
