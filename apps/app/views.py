# -*- encoding: utf-8 -*-
import glob
from django import template
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.shortcuts import redirect, render
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
import os
import time
from multiprocessing import Pool, cpu_count, Array

from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from pathlib import Path

from .models import SAGRAData
from zipfile import ZipFile

df3 = None


@login_required(login_url="/login/")
def index(request):

    if request.POST.get('validation-switcher'):
        switch = True
    else:
        switch = False

    filename = ''

    data = {
        'Informações': 'Informações',
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

    html_template = loader.get_template('index.html')

    if request.FILES.get('myfile', False):
        SAGRAData.objects.all().delete()

        # SAGRAData.objects.filter(pk=1).delete()

        myfile = request.FILES['myfile']

        item_value = request.POST.get('item_value')
        selected_days = request.POST.get('selectedDays')

        fs = FileSystemStorage()
        filename = fs.save('core/static/files/upload/' + myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        uploaded_file_url, period_dates = open_file_automodel(
            myfile.name, item_value, selected_days, switch)

        context = {
            'data': data,
            '\  ': uploaded_file_url,
            'series': True,
            'filename': myfile.name,
            'data_json': uploaded_file_url[0:5],
            "period_dates": period_dates
        }
    else:
        if request.method == 'POST':
            item_value = request.POST.get('item_value')
            selected_days = request.POST.get('selectedDays')
            data_json, period_dates = open_file_automodel(
                filename, item_value, selected_days, switch)

            data_json = json.dumps(str(data_json))

            context = {
                'data': data,  'series': True,
                'data_json': json.loads(data_json),
                'filename': filename,
                "period_dates": period_dates
            }
        else:
            context = {'data': data,  'series': False}

    return HttpResponse(html_template.render(context, request))


@ login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))

        context['segment'] = load_template

        html_template = loader.get_template('page-400.html')

        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('page-500.html')
        return HttpResponse(html_template.render(context, request))


def open_file_automodel(filename, item_value, periods, switch):

    file_path = Path(filename)
    file_extension = file_path.suffix.lower()[1:]

    if SAGRAData.objects.exists():
        df3 = pd.DataFrame.from_records(SAGRAData.objects.all().values())

    else:
        SAGRAData.objects.all().delete()

        df3 = pd.read_excel(f'core/static/files/upload/{filename}')

        if file_extension == 'xlsx':
            df3 = pd.read_excel(
                f'core/static/files/upload/{filename}', engine='openpyxl')

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
        }, inplace=True, errors='raise')

        host = settings.DATABASES['default']['HOST']
        user = settings.DATABASES['default']['USER']
        password = settings.DATABASES['default']['PASSWORD']
        database_name = settings.DATABASES['default']['NAME']

        # database_url = 'postgresql://{user}:{password}@{host}:5432/{database_name}'.format(
        #     host=host,
        #     user=user,
        #     password=password,
        #     database_name=database_name,
        # )

        # engine = create_engine(database_url, echo=False)

        engine = create_engine('sqlite:///db.sqlite3')

        df3.to_sql(SAGRAData._meta.db_table,
                   if_exists='replace', con=engine, index_label='id', index=True)

        # engine = create_engine('sqlite:///db.sqlite3')

        # df3.to_sql(SAGRAData._meta.db_table,
        #            if_exists='replace', con=engine, index_label='id', index=True)

    field = item_value
    n_periods = int(periods)

    start_test = (df3['date_occurrence'][0]).strftime("%d-%m-%Y")
    end_test = (df3['date_occurrence'][len(df3)-1]).strftime("%d-%m-%Y")

    print(f"Data inicio {start_test}")
    print(f"Data fim {end_test}")

    period_dates = {
        "start_date": {start_test},
        "end_date": {end_test},
    }

    # df3.Data = pd.to_datetime(df3['date_occurrence'])
    df3.set_index('date_occurrence', inplace=True)
    # df3 = df3.loc[:, ~df3.columns.str.contains('^id')]
    df3.sort_index(inplace=True)
    df3.head()
    plt.figure(figsize=(15, 6))
    plt.grid()
    plt.tight_layout()
    plt.title(
        f'Período entre datas: {start_test} - {end_test}')
    plt.xlabel("Date")
    plt.ylabel(field.replace('_', ' ').capitalize())
    plt.tight_layout()
    plt.plot(df3.index, df3[field], label='linear')
    plt.savefig(
        "core/static/files/grafico_total_periodo.png",
        dpi=300, bbox_inches='tight'
    )

    automodel = model_auto_ARIMA(df3[field], switch)

    data = plotarima(n_periods, automodel, df3, field)

    # remove all excell files
    files = glob.glob('core/static/files/upload/*.xls*')
    for f in files:
        os.remove(f)

    zip_files()

    return data, period_dates


def plotarima(n_periods, automodel, serie, field):

    # Forecast
    fc, confint = automodel.predict(
        n_periods=n_periods, return_conf_int=True)

    # print(
    #     f'############################### FC: {fc} confint: {confint} \
    #     --#######- {serie.index[serie.shape[0]-1]}')

    # Weekly index
    # fc_ind = pd.date_range(timeseries.index[0],{{
    # periods=n_periods, freq="D")}}
    fc_ind = pd.date_range(serie.index[serie.shape[0]-1],
                           periods=n_periods, freq="D")

    # Calendar index
    # fc_ind = pd.date_range(
    #     start=serie.index[serie.shape[0]-1], end='2020-10-31 00:00:00+00:00')
    # print(fc_ind)

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
        data_dict[str(datetime.fromtimestamp(int(key[:-3])))
                  [:10]] = str(round(value, 2))

    data = json.dumps(data_dict, indent=4)

    # print('TESTE: ', data)
    # Upper and lower confidence bounds
    lower_series = pd.Series(confint[:, 0], index=fc_ind)
    upper_series = pd.Series(confint[:, 1], index=fc_ind)

    # lower_series = lower_series.to_json(orient='index')

    series = serie[field]

    # print('#### SERIES ####', series)

    data_series = series.to_json(orient='index')

    data_series = json.loads(data_series)

    data_serie = dict()

    for key, value in data_series.items():
        # print(str(datetime.fromtimestamp(
        #     int(key[:-3])))[:10], '->', str(value))
        data_serie[str(datetime.fromtimestamp(int(key[:-3])))
                   [:10]] = str(value)

    data_serie = json.dumps(data_serie)

    # print(
    #     f'#####$$$$$$$$$$$$$$$$$$$$$$$$$$$$ lower_series.index: {data_serie}')

    period = json.loads(data)

    # Create plot
    plt.figure(figsize=(15, 6))
    plt.grid()
    # plt.tight_layout()
    plt.plot(serie[field])
    plt.plot(fc_series, color="orange")
    plt.title(
        f'Período de predição {n_periods} dias: {list(period.items())[0][0]} - {list(period.items())[-1][0]}')
    plt.xlabel("Date")
    plt.ylabel(serie[field].name.replace('_', ' ').capitalize())
    plt.fill_between(lower_series.index, lower_series, upper_series, color="k",
                     alpha=0.15)
    plt.legend(("past", "forecast", "95% confidence interval"),
               loc="upper left")
    plt.tight_layout()
    plt.savefig("core/static/files/predicao.png",
                dpi=300, bbox_inches='tight')
    # plt.show()

    json_list = []

    # print(
    #     f'#####$$$$$$$$$$$$$$$$$$$$$$$$$$$$ DATA: {list(period.items())[0][0]}-- {list(period.items())[-1][0]}')

    jsonMerged = {**json.loads(data_serie), **json.loads(data)}

    create_xlsx(jsonMerged)

    for key, value in jsonMerged.items():
        json_list.append({"y": key, "a": value})

    # data = json.dumps(jsonMerged)
    # print(data)

    # data = {"data_json": data}

    # print('JASON FORMAT: ', json_list)
    # print('JASON FORMAT: ', str(json_list).replace('[', '').replace(']', ''))

    # context = {}

    # context["data"] = ','.join([str(i) for i in json_list])

    # print(json.dumps(context))

    # html_template = loader.get_template('index.html')

    # return HttpResponse(html_template.render(context))

    return json_list


def timed(func):
    def _wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f"Completed {func.__name__} in {time.time() - start:.3f} sec")
        return res
    return _wrapper


@timed
def model_auto_ARIMA(df, switch):
    if switch:
        D = 1
    else:
        D = 0

    model = auto_arima(
        df, start_p=1, start_q=1,
        test='adf',       # use adftest to find optimal 'd'
        seasonal=switch,   # No Seasonality
        start_P=0,
        m=12,
        D=D,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        simple_differencing=True,
    )

    print(model.summary)

    return model


def create_xlsx(data):
    df = pd.DataFrame(data=data, index=[0])
    df = (df.T)
    df.to_excel('core/static/files/predicao.xlsx')


def zip_files():
    # create a ZipFile object
    zipObj = ZipFile('core/static/files/predicao.zip', 'w')
    # Add multiple files to the zip
    zipObj.write('core/static/files/grafico_total_periodo.png')
    zipObj.write('core/static/files/predicao.png')
    zipObj.write('core/static/files/predicao.xlsx')
    # close the Zip File
    zipObj.close()
