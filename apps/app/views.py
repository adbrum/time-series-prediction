# -*- encoding: utf-8 -*-
from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.shortcuts import redirect, render
# from predictions.forms import DocumentForm
# from predictions.models import SAGRAData
from pmdarima import auto_arima
from datetime import datetime
from pandas import concat
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from math import sqrt
from sqlalchemy import create_engine
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import json

from apps.predictions.models import SAGRAData


@login_required(login_url="/login/")
def index(request):
    # context = {'segment': 'index'}

    data = {
        '': '',
        'Tmed (ºC)': 'average_temperature',
        'Tmax (ºC)': 'maximum_temperature',
        'Tmin (ºC)': 'minimum_temperature',
        'HRmed (%)': 'average_humidity',
        'HRmax (%)': 'maximum_humidity',
        'HRmin (%)': 'minimum_humidity',
        'RSG (kj/m2)': 'RSG',
        'DV (graus)': 'DV',
        'VVmed (m/s)': 'average_wind_speed',
        'VVmax (m/s)': 'maximum_wind_speed',
        'P (mm)': 'rainfall',
        'Tmed Relva(ºC)': 'average_grass_temperature',
        'Tmax Relva(ºC)': 'maximum_grass_temperature',
        'Tmin Relva(ºC)': 'minimum_grass_temperature',
        'ET0 (mm)': 'ET0',
    }
    context = {'data': data}

    html_template = loader.get_template('index.html')

    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']

        item_value = request.POST.get('item_value')

        # fs = FileSystemStorage()
        # filename = fs.save('assets/' + myfile.name, myfile)
        # uploaded_file_url = fs.url(filename)
        uploaded_file_url = open_file_automodel(myfile.name, item_value)

        context = {'data': data,  '\  ': uploaded_file_url, 'series': True}

    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))

        context['segment'] = load_template

        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('page-500.html')
        return HttpResponse(html_template.render(context, request))


def open_file_automodel(filename, item_value):

    df3 = pd.read_excel(f'core/static/files/{filename}')
    # df3.insert(0, 'id', None)

    df3.rename(columns={
        'EMA': 'EMA',
        'Data': 'date_occurrence',
        'Tmed (ºC)': 'average_temperature',
        'Tmax (ºC)': 'maximum_temperature',
        'Tmin (ºC)': 'minimum_temperature',
        'HRmed (%)': 'average_humidity',
        'HRmax (%)': 'maximum_humidity',
        'HRmin (%)': 'minimum_humidity',
        'RSG (kj/m2)': 'RSG',
        'DV (graus)': 'DV',
        'VVmed (m/s)': 'average_wind_speed',
        'VVmax (m/s)': 'maximum_wind_speed',
        'P (mm)': 'rainfall',
        'Tmed Relva(ºC)': 'average_grass_temperature',
        'Tmax Relva(ºC)': 'maximum_grass_temperature',
        'Tmin Relva(ºC)': 'minimum_grass_temperature',
        'ET0 (mm)': 'ET0',
    },
        inplace=True, errors='raise')

    engine = create_engine('sqlite:///db.sqlite3')

    df3.to_sql(SAGRAData._meta.db_table,
               if_exists='replace', con=engine, index_label='id', index=True)

    field = item_value
    n_periods = 60

    start_test = (df3['date_occurrence'][0]).strftime("%d-%m-%Y")
    end_test = (df3['date_occurrence'][len(df3)-1]).strftime("%d-%m-%Y")

    print(f"Data inicio {start_test}")
    print(f"Data fim {end_test}")

    df3.Data = pd.to_datetime(df3['date_occurrence'])
    df3.set_index('date_occurrence', inplace=True)
    df3.sort_index(inplace=True)
    df3.head()
    plt.figure(figsize=(15, 6))
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Date")
    plt.ylabel(field)
    plt.tight_layout()
    plt.plot(df3.index, df3[field], label='linear')
    plt.savefig(
        "core/static/files/autoarima_01.png",
        dpi=300, bbox_inches='tight'
    )
    # plt.show()

    # automodel = arimamodel(timeseries)
    data_order, automodel = model_auto_ARIMA(df3, field)

    data = plotarima(n_periods, automodel, df3, field)

    return data


def plotarima(n_periods, automodel, serie, field):
    # Forecast
    fc, confint = automodel.predict(n_periods=n_periods, return_conf_int=True)

    # print(
    # f'############################### FC: {fc} confint: {confint} \
    #     --#######- {serie.index[serie.shape[0]-1]}')

    # Weekly index
    # fc_ind = pd.date_range(timeseries.index[0],{{
    # periods=n_periods, freq="D")}}
    fc_ind = pd.date_range(serie.index[serie.shape[0]-1],
                           periods=n_periods, freq="D")
    # print(dict(zip(fc_ind[0], fc)))

    # Forecast series
    fc_series = pd.Series(fc, index=fc_ind)
    json_serie = fc_series.to_json(orient='index')

    # print(
    # f'#####$$$$$$$$$$$$$$$$$$$$$$$$$$$$ fc_series: {json_serie}')

    data = json.loads(json_serie)

    data_dict = dict()

    for key, value in data.items():
        # print(str(datetime.fromtimestamp(int(key[:-3]))), '->', str(value))
        data_dict[str(datetime.fromtimestamp(int(key[:-3])))] = str(value)

    data = json.dumps(data_dict, indent=4)

    # print('TESTE: ', data)
    # Upper and lower confidence bounds
    lower_series = pd.Series(confint[:, 0], index=fc_ind)
    upper_series = pd.Series(confint[:, 1], index=fc_ind)

    # lower_series = lower_series.to_json(orient='index')

    series = serie[field]

    data_series = series.to_json(orient='index')

    data_series = json.loads(data_series)

    data_serie = dict()

    for key, value in data_series.items():
        # print(str(datetime.fromtimestamp(int(key[:-3]))), '->', str(value))
        data_serie[str(datetime.fromtimestamp(int(key[:-3])))] = str(value)

    data_serie = json.dumps(data_serie, indent=4)

    # print(
    # f'#####$$$$$$$$$$$$$$$$$$$$$$$$$$$$ lower_series.index: {data_serie}')

    # Create plot
    plt.figure(figsize=(15, 6))
    plt.grid()
    # plt.tight_layout()
    plt.plot(serie[field])
    plt.plot(fc_series, color="red")
    plt.xlabel("Date")
    plt.ylabel(serie[field].name)
    plt.fill_between(lower_series.index, lower_series, upper_series, color="k",
                     alpha=0.25)
    plt.legend(("past", "forecast", "95% confidence interval"),
               loc="upper left")
    plt.tight_layout()
    plt.savefig("core/static/files/autoarima.png",
                dpi=300, bbox_inches='tight')
    # plt.show()

    jsonMerged = {**json.loads(data_serie), **json.loads(data)}
    data = json.dumps(jsonMerged)
    print(data)

    context = {'data': data}

    html_template = loader.get_template('index.html')

    html_template = loader.get_template('index.html')

    return HttpResponse(html_template.render(context))


def model_auto_ARIMA(df, field):
    model = auto_arima(df[field], start_p=0, start_q=0,
                       test='adf',       # use adftest to find optimal 'd'
                       max_p=4, max_q=4,  # maximum p and q
                       m=12,              # frequency of series
                       d=None,           # let model determine 'd'
                       seasonal=False,   # No Seasonality
                       start_P=0,
                       D=1,
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=False)

    model.summary()
    print(model.aic())

    get_parametes = model.get_params()

    order = get_parametes.get('order')
    seasonal_order = get_parametes.get('seasonal_order')

    data_order = {
        "order": order,
        "seasonal_order": seasonal_order
    }

    return data_order, model
