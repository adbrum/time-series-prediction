# -*- encoding: utf-8 -*-
import glob
import json
import os
import time
from datetime import datetime
from math import sqrt
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.template import loader
from pmdarima import auto_arima
from sqlalchemy import create_engine
from django.shortcuts import render, redirect
from django.template import TemplateDoesNotExist
from pandas import read_excel
from functools import wraps
from matplotlib import pyplot as plt

from .models import SAGRAData

df = None

names = {
    "average_temperature": "Temperatura Média",
    "maximum_temperature": "Temperatura Máxima",
    "minimum_temperature": "Temperatura Mínima",
    "average_humidity": "Humidade Média",
    "maximum_humidity": "Humidade Máxima",
    "minimum_humidity": "Humidade Mínima",
    "RSG": "Radiação Solar Global",
    "DV": "Difusão de Vento",
    "average_wind_speed": "Velocidade Média do Vento",
    "maximum_wind_speed": "Velocidade Máxima do Vento",
    "rainfall": "Pluviosidade",
    "average_grass_temperature": "Temperatura Média da Relva",
    "maximum_grass_temperature": "Temperatura Máxima da Relva",
    "minimum_grass_temperature": "Temperatura Mínima da Relva",
    "ET0": "Evapotranspiração",
}


@login_required(login_url="/login/")
def index(request):
    switch = bool(request.POST.get("validation-switcher"))
    filename = ""

    data = {
        "Informações": "Informações",
        "Tmed (ºC)": "average_temperature",
        "Tmax (ºC)": "maximum_temperature",
        "Tmin (ºC)": "minimum_temperature",
        "HRmed (%)": "average_humidity",
        "HRmax (%)": "maximum_humidity",
        "HRmin (%)": "minimum_humidity",
        "RSG (kj/m2)": "RSG",
        "DV (graus)": "DV",
        "VVmed (m/s)": "average_wind_speed",
        "VVmax (m/s)": "maximum_wind_speed",
        "P (mm)": "rainfall",
        "Tmed Relva(ºC)": "average_grass_temperature",
        "Tmax Relva(ºC)": "maximum_grass_temperature",
        "Tmin Relva(ºC)": "minimum_grass_temperature",
        "ET0 (mm)": "ET0",
    }

    html_template = loader.get_template("index.html")

    if request.FILES.get("myfile", False):
        SAGRAData.objects.all().delete()

        myfile = request.FILES["myfile"]
        item_value = request.POST.get("item_value")
        selected_days = request.POST.get("selectedDays")

        fs = FileSystemStorage()
        filename = fs.save(f"core/static/files/upload/{myfile.name}", myfile)
        uploaded_file_url = fs.url(filename)

        uploaded_file_url, period_dates = open_file_automodel(
            myfile.name, item_value, selected_days, switch
        )

        context = {
            "data": data,
            "uploaded_file_url": uploaded_file_url,
            "series": True,
            "filename": myfile.name,
            "data_json": uploaded_file_url[:5],
            "period_dates": period_dates,
        }

    elif request.method == "POST":
        item_value = request.POST.get("item_value")
        selected_days = request.POST.get("selectedDays")
        data_json, period_dates = open_file_automodel(
            filename, item_value, selected_days, switch
        )

        data_json = json.dumps(str(data_json))

        context = {
            "data": data,
            "series": True,
            "data_json": json.loads(data_json),
            "filename": filename,
            "period_dates": period_dates,
        }
    else:
        context = {"data": data, "series": False}

    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    try:
        load_template = request.path.split("/")[-1]

        if load_template == "admin":
            return redirect("admin:index")

        context = {"segment": load_template}
        html_template = get_html_template(load_template)

    except TemplateDoesNotExist:
        html_template = get_html_template("page-404")

    except Exception:
        html_template = get_html_template("page-500")

    return render(request, html_template, context)


def get_html_template(template_name):
    return f"page-{template_name}.html"


def open_file_automodel(filename, item_value, periods, switch):
    file_path = Path(filename)
    file_extension = file_path.suffix.lower()[1:]

    if SAGRAData.objects.exists():
        print("There are data already loaded")
        df = pd.DataFrame.from_records(SAGRAData.objects.all().values())
    else:
        df = open_file(filename, file_extension)

    field = item_value
    n_periods = int(periods)

    start_test = df["date_occurrence"].iloc[0].strftime("%d-%m-%Y")
    end_test = df["date_occurrence"].iloc[-1].strftime("%d-%m-%Y")

    print(f"Data inicio {start_test}")
    print(f"Data fim {end_test}")

    period_dates = {
        "start_date": start_test,
        "end_date": end_test,
    }

    df = df.sort_values("date_occurrence", ascending=False)
    df = df.drop_duplicates(subset="date_occurrence", keep="first")
    df.set_index("date_occurrence", inplace=True)
    df.sort_index(inplace=True)
    df.head()

    plt.figure(figsize=(15, 6))
    plt.grid()
    plt.tight_layout()
    plt.title(f"Período entre datas: {start_test} - {end_test}")
    plt.xlabel("Data")
    plt.ylabel(names[field])
    plt.tight_layout()
    plt.plot(df.index, df[field], label="linear")
    plt.savefig(
        "core/static/files/grafico_total_periodo.png", dpi=300, bbox_inches="tight"
    )

    automodel = model_auto_ARIMA(df[field], switch)

    data = plot_arima(n_periods, automodel, df, field)

    cleanup_files()

    return data, period_dates


def cleanup_files():
    # Remove all excel files
    files = glob.glob("core/static/files/upload/*.xls*")
    for f in files:
        os.remove(f)


def open_file(filename, file_extension):
    SAGRAData.objects.all().delete()

    file_path = f"core/static/files/upload/{filename}"

    if file_extension == "xlsx":
        result = read_excel(file_path, engine="openpyxl")
    else:
        result = read_excel(file_path)

    result = rename_columns(result)

    engine = create_engine("sqlite:///db.sqlite3")

    result.to_sql(
        SAGRAData._meta.db_table,
        con=engine,
        if_exists="replace",
        index_label="id",
        index=True,
    )

    return result


def rename_columns(dataframe):
    renamed_columns = {
        "EMA": "EMA",
        "Data": "date_occurrence",
        "Tmed (ºC)": "average_temperature",
        "Tmax (ºC)": "maximum_temperature",
        "Tmin (ºC)": "minimum_temperature",
        "HRmed (%)": "average_humidity",
        "HRmax (%)": "maximum_humidity",
        "HRmin (%)": "minimum_humidity",
        "RSG (kj/m2)": "RSG",
        "DV (graus)": "DV",
        "VVmed (m/s)": "average_wind_speed",
        "VVmax (m/s)": "maximum_wind_speed",
        "P (mm)": "rainfall",
        "Tmed Relva(ºC)": "average_grass_temperature",
        "Tmax Relva(ºC)": "maximum_grass_temperature",
        "Tmin Relva(ºC)": "minimum_grass_temperature",
        "ET0 (mm)": "ET0",
    }

    dataframe.rename(columns=renamed_columns, inplace=True, errors="raise")

    return dataframe


def plot_arima(n_periods, automodel, serie, field):
    # Forecast
    fc, confint = automodel.predict(n_periods=n_periods, return_conf_int=True)

    fc_ind = pd.date_range(serie.index[-1], periods=n_periods, freq="D")

    # Forecast series
    fc_series = pd.Series(fc, index=fc_ind)
    json_serie = fc_series.to_json(orient="index")

    data = json.loads(json_serie)
    data_dict = {}

    for key, value in data.items():
        forecast_date = datetime.fromtimestamp(int(key[:-3])).strftime("%Y-%m-%d")
        data_dict[forecast_date] = round(value, 2)

    data = json.dumps(data_dict, indent=4)

    # Upper and lower confidence bounds
    lower_series = pd.Series(confint[:, 0], index=fc_ind)
    upper_series = pd.Series(confint[:, 1], index=fc_ind)

    data_series = serie[field].to_json(orient="index")
    data_series = json.loads(data_series)

    data_serie = {}

    for key, value in data_series.items():
        data_date = datetime.fromtimestamp(int(key[:-3])).strftime("%Y-%m-%d")
        data_serie[data_date] = value

    data_serie = json.dumps(data_serie)

    period = json.loads(data)

    # Create plot
    plt.figure(figsize=(15, 6))
    plt.grid()
    plt.plot(serie[field])
    plt.plot(fc_series, color="orange")
    plt.title(
        f"Período de predição {n_periods} dias: {list(period.items())[0][0]} - {list(period.items())[-1][0]}"
    )
    plt.xlabel("Data")
    plt.ylabel(names[field])
    plt.fill_between(
        lower_series.index, lower_series, upper_series, color="k", alpha=0.15
    )
    plt.legend(
        ("dados anteriores", "predição", "95% intervalo de confiança"), loc="upper left"
    )
    plt.tight_layout()
    plt.savefig("core/static/files/predicao.png", dpi=300, bbox_inches="tight")

    jsonMerged = {**json.loads(data_serie), **json.loads(data)}
    create_xlsx(jsonMerged)

    return []


def create_xlsx(data_dict):
    df = pd.DataFrame(data_dict.items(), columns=["Date", "Value"])
    filename = "core/static/files/prediction.xlsx"
    df.to_excel(filename, index=False)


def timed(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Completed {func.__name__} in {end - start:.3f} seconds")
        return res

    return _wrapper


import time
from functools import wraps


def timed(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Completed {func.__name__} in {end - start:.3f} seconds")
        return res

    return _wrapper


@timed
def model_auto_ARIMA(df, seasonal):
    D = 1 if seasonal else 0

    model = auto_arima(
        df,
        start_p=1,
        d=None,
        start_q=1,
        max_p=3,
        max_d=2,
        max_q=3,
        start_P=0,
        D=D,
        seasonal=seasonal,
        m=12,
        error_action="warn",
        trace=True,
        random_state=42,
        n_fits=20,
        suppress_warnings=True,
        stepwise=True,
        information_criterion="aic",
        alpha=0.05,
    )

    print(model.summary())
    print(f"Best ARIMA {model.order}x{model.seasonal_order} model")
    print(f"AIC: {model.aic()}")
    print(f"BIC: {model.bic()}")

    return model


def zip_files():
    # create a ZipFile object
    zipObj = ZipFile("core/static/files/predicao.zip", "w")
    # Add multiple files to the zip
    zipObj.write("core/static/files/grafico_total_periodo.png")
    zipObj.write("core/static/files/predicao.png")
    zipObj.write("core/static/files/predicao.xlsx")
    # close the Zip File
    zipObj.close()
