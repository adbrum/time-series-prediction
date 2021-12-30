# -*- encoding: utf-8 -*-
import glob
import json
import os
import time
from datetime import datetime
from math import sqrt
from multiprocessing import Pool, cpu_count
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
from django import template
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import redirect, render
from django.template import loader
from django.urls import reverse
from joblib import delayed
from pandas import concat
from pandas.plotting import autocorrelation_plot
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

from .models import SAGRAData

df = None

names = {
    'average_temperature': 'Temperatura Média',
    'maximum_temperature': 'Temperatura Máxima',
    'minimum_temperature': 'Temperatura Mínima',
    'average_humidity': 'Humidade Média',
    'maximum_humidity': 'Humidade Máxima',
    'minimum_humidity': 'Humidade Mínima',
    'RSG': 'Radiação Solar Global',
    'DV': 'Difusão de Vento',
    'average_wind_speed': 'Velocidade Média do Vento',
    'maximum_wind_speed': 'Velocidade Máxima do Vento',
    'rainfall': 'Pluviosidade',
    'average_grass_temperature': 'Temperatura Média da Relva',
    'maximum_grass_temperature': 'Temperatura Máxima da Relva',
    'minimum_grass_temperature': 'Temperatura Mínima da Relva',
    'ET0': 'Evapotranspiração',
}


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


@login_required(login_url="/login/")
def pages(request):
    context = {}

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

        df = pd.DataFrame.from_records(SAGRAData.objects.all().values())

    else:
        SAGRAData.objects.all().delete()

        df = pd.read_excel(f'core/static/files/upload/{filename}')

        if file_extension == 'xlsx':
            df = pd.read_excel(
                f'core/static/files/upload/{filename}', engine='openpyxl')

        df.rename(columns={
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

        # host = settings.DATABASES['default']['HOST']
        # user = settings.DATABASES['default']['USER']
        # password = settings.DATABASES['default']['PASSWORD']
        # database_name = settings.DATABASES['default']['NAME']

        # database_url = 'postgresql://{user}:{password}@{host}:5432/{database_name}'.format(
        #     host=host,
        #     user=user,
        #     password=password,
        #     database_name=database_name,
        # )

        # engine = create_engine(database_url, echo=False)

        # df.to_sql(SAGRAData._meta.db_table,
        #            if_exists='replace', con=engine, index_label='id', index=True)

        engine = create_engine('sqlite:///db.sqlite3')

        df.to_sql(SAGRAData._meta.db_table,
                  if_exists='replace', con=engine, index_label='id', index=True)

    field = item_value
    n_periods = int(periods)

    start_test = (df['date_occurrence'][0]).strftime("%d-%m-%Y")
    end_test = (df['date_occurrence'][len(df)-1]).strftime("%d-%m-%Y")

    print(f"Data inicio {start_test}")
    print(f"Data fim {end_test}")

    period_dates = {
        "start_date": {start_test},
        "end_date": {end_test},
    }

    df = df.sort_values('date_occurrence', ascending=False)
    df = df.drop_duplicates(subset='date_occurrence', keep='first')
    # df.Data = pd.to_datetime(df['date_occurrence'])
    df.set_index('date_occurrence', inplace=True)
    # df = df.loc[:, ~df.columns.str.contains('^id')]
    df.sort_index(inplace=True)
    df.head()
    plt.figure(figsize=(15, 6))
    plt.grid()
    plt.tight_layout()
    plt.title(
        f'Período entre datas: {start_test} - {end_test}')
    plt.xlabel("Data")
    plt.ylabel(names[field])
    plt.tight_layout()
    plt.plot(df.index, df[field], label='linear')
    plt.savefig(
        "core/static/files/grafico_total_periodo.png",
        dpi=300, bbox_inches='tight'
    )

    automodel = model_auto_ARIMA(df[field], switch)

    data = plotarima(n_periods, automodel, df, field)

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

    fc_ind = pd.date_range(serie.index[serie.shape[0]-1],
                           periods=n_periods, freq="D")

    # Forecast series
    fc_series = pd.Series(fc, index=fc_ind)
    json_serie = fc_series.to_json(orient='index')

    data = json.loads(json_serie)

    data_dict = dict()

    for key, value in data.items():
        # print(str(datetime.fromtimestamp(int(key[:-3]))), '->', str(value))
        data_dict[str(datetime.fromtimestamp(int(key[:-3])))
                  [:10]] = str(round(value, 2))

    data = json.dumps(data_dict, indent=4)

    # Upper and lower confidence bounds
    lower_series = pd.Series(confint[:, 0], index=fc_ind)
    upper_series = pd.Series(confint[:, 1], index=fc_ind)

    series = serie[field]

    data_series = series.to_json(orient='index')

    data_series = json.loads(data_series)

    data_serie = dict()

    for key, value in data_series.items():
        # print(str(datetime.fromtimestamp(
        #     int(key[:-3])))[:10], '->', str(value))
        data_serie[str(datetime.fromtimestamp(int(key[:-3])))
                   [:10]] = str(value)

    data_serie = json.dumps(data_serie)

    period = json.loads(data)

    # Create plot
    plt.figure(figsize=(15, 6))
    plt.grid()
    # plt.tight_layout()
    plt.plot(serie[field])
    plt.plot(fc_series, color="orange")
    plt.title(
        f'Período de predição {n_periods} dias: {list(period.items())[0][0]} - {list(period.items())[-1][0]}')
    plt.xlabel("Data")
    plt.ylabel(names[field])
    plt.fill_between(lower_series.index, lower_series, upper_series, color="k",
                     alpha=0.15)
    plt.legend(("dados anteriores", "predição", "95% intervalo de confiança"),
               loc="upper left")
    plt.tight_layout()
    plt.savefig("core/static/files/predicao.png",
                dpi=300, bbox_inches='tight')

    json_list = []

    jsonMerged = {**json.loads(data_serie), **json.loads(data)}

    create_xlsx(jsonMerged)

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
        max_p=3, max_q=3, m=12,
        start_P=0, seasonal=switch,
        test='adf', D=D, trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        simple_differencing=True,
    )

    print(model.summary())
    print(f"Best ARIMA {model.order} model")
    print(f"AIC: {model.aic()}")
    print(f"BIC: {model.bic()}")
    print(f"AICc: {model.aicc()}")

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
