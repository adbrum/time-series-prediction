
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
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import json


def home(request):
    documents = SAGRAData.objects.all()
    return render(request, 'index.html', {'documents': documents})


def simple_upload(request):

    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']

        # fs = FileSystemStorage()
        # filename = fs.save('assets/' + myfile.name, myfile)
        # uploaded_file_url = fs.url(filename)
        uploaded_file_url = open_file(myfile.name)
        return render(request, 'simple_upload_prediction.html', {
            '\  ': uploaded_file_url
        })

    return render(request, 'simple_upload.html')


def simple_upload_prediction(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']

        # fs = FileSystemStorage()
        # filename = fs.save('assets/' + myfile.name, myfile)
        # uploaded_file_url = fs.url(filename)
        uploaded_file_url = open_file_prediction(myfile.name)
        return render(request, 'index.html', {
            '\  ': uploaded_file_url
        })

    return render(request, 'simple_upload_prediction.html')


def automodel_prediction(request):
    items = [
        'Tmed (ºC)',
        'Tmax (ºC)',
        'Tmin (ºC)',
        'HRmed ( % )',
        'HRmax (%)',
        'HRmin (%)',
        'RSG (kj/m2)',
        'DV (graus)',
        'VVmed (m/s)',
        'VVmax (m/s)',
        'P (mm)',
        'Tmed Relva(ºC)',
        'Tmax Relva(ºC)',
        'Tmin Relva(ºC)',
        'ET0 (mm)'
    ]
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']

        item_value = 'Tmin (ºC)'  # request.POST.get('item_value')
        print('\n================================ ', item_value)

        # fs = FileSystemStorage()
        # filename = fs.save('assets/' + myfile.name, myfile)
        # uploaded_file_url = fs.url(filename)
        uploaded_file_url = open_file_automodel(myfile.name, item_value)
        return render(request, 'charts-morris.html', {
            '\  ': uploaded_file_url
        })

    return render(request, 'index.html', {'items': items})


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'model_form_upload.html', {
        'form': form
    })


def between_dates(df):

    start = (df['Date'][0]).strftime("%d-%m-%Y")
    end = (df['Date'][len(df)-1]).strftime("%d-%m-%Y")

    return start, end


def open_file(filename):
    global df
    df = pd.read_excel(f'assets/{filename}')

    start, end = between_dates(df)

    global field
    field = 'Tmin (ºC)'

    df.Date = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df[field], label='linear')
    plt.legend(("Data inicio " + start + " -" + " Data fim " + end, ),
               loc="upper left")
    plt.savefig(
        "assets/01.png",
        dpi=300, bbox_inches='tight'
    )
    # plt.show()

    result_decompose(df, field)

    data_order = model_auto_ARIMA(df, field)

    test, training = train_and_test(df, field, data_order)

    train_test_predction(test, training, data_order)


# salvar a decomposicao em result
def result_decompose(df, field):
    result = seasonal_decompose(df[field], model='aditive', period=12)

    # plotar os 4 gráficos
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
    result.observed.plot(ax=ax1)
    result.trend.plot(ax=ax2)
    result.seasonal.plot(ax=ax3)
    result.resid.plot(ax=ax4)
    # plt.figure(figsize=(15, 6))
    plt.tight_layout()
    plt.savefig(
        "assets/02.png",
        dpi=300, bbox_inches='tight'
    )
    series = df.loc[:, field].values

    print(adf_test(series))

    print(kpss_test(series))


def open_file_prediction(filename):
    field = 'Tmin (ºC)'

    # global df2
    df2 = pd.read_excel(f'assets/{filename}')

    start, end = between_dates(df2)

    start_test = (df2['Date'][0]).strftime("%d-%m-%Y")

    end_test = (df2['Date'][len(df2)-1]).strftime("%d-%m-%Y")
    print(f"Data inicio {start_test}")
    print(f"Data fim {end_test}")
    df2.Date = pd.to_datetime(df2.Date)
    df2.set_index('Date', inplace=True)
    df2.sort_index(inplace=True)
    df2.head()
    plt.figure(figsize=(15, 6))
    plt.tight_layout()
    plt.plot(df2.index, df2[field], label='linear')
    plt.legend(("Data inicio " + start + " -" +
               " Data fim " + end, ), loc="upper left")
    plt.savefig(
        "assets/04.png",
        dpi=300, bbox_inches='tight'
    )
    plt.show()

    data_order = model_auto_ARIMA(df2, field)

    prediction_test(data_order, df2)

    predicton_SARIMAX(data_order, df2)


# ADF Test


def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')

# KPSS Test


def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(
        f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')


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


def train_and_test(df, field, data_order):

    print('order:', data_order['order'])
    print('seasonal_order:', data_order['seasonal_order'])

    model = model_SARIMAX(df[field], data_order)
    model_fit = model.fit()
    model_fit.summary()

    # split into train and test sets
    X = df[field].values  # anos 1-1-2015 - 31-12-2019
    training_size = int(len(X) * 0.70)
    training, test = X[0:training_size], X[training_size:len(X)]

    training_start = (df.index[0]).strftime("%d-%m-%Y")
    end_training = (df.index[len(training)-1]).strftime("%d-%m-%Y")

    testing_start = (df.index[training_size]).strftime("%d-%m-%Y")
    end_testing = (df.index[len(X)-1]).strftime("%d-%m-%Y")

    print(f"Dados de treino: \ninicio: {training_start}, fim: {end_training}")
    print(f"Dados de teste: \ninicio: {testing_start}, fim: {end_testing}")

    res = model.fit(disp=False)
    print(res.summary())
    return test, training


def model_SARIMAX(df, data_order):
    model = SARIMAX(df,
                    order=(data_order['order'][0],
                           data_order['order'][1], data_order['order'][2]),
                    seasonal_order=(data_order['seasonal_order'][0],
                                    data_order['seasonal_order'][1],
                                    data_order['seasonal_order'][2],
                                    data_order['seasonal_order'][3]),
                    suppress_warnings=True,
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    return model


def train_test_predction(test, training, data_order):

    history = [x for x in training]
    predictions = []
    # walk-forward validation
    for t in range(len(test)):
        model = model_SARIMAX(history, data_order)
        model_fit = model.fit()
        output = model_fit.forecast()
        predicted_value = output[0]
        predictions.append(predicted_value)
        obs = test[t]
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))

    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    image_test_prediction('03', test, predictions)

    # plt.show()
    return model, predictions


def image_test_prediction(name, test, predictions):
    plt.figure(figsize=(15, 6))

    # plt.title(field + f' em Beja de {training_start} a {end_testing}')
    # plot forecasts against actual outcomes
    plt.plot(test, color='orange')
    plt.plot(predictions, color='blue')
    plt.legend(["Teste", "Predição"], loc="lower right")
    plt.xlabel('Dias')
    plt.ylabel('Temperatura')
    plt.tight_layout()
    plt.savefig(
        f"assets/{name}.png",
        dpi=300, bbox_inches='tight'
    )


def open_file_automodel(filename, item_value):

    df3 = pd.read_excel(f'assets/{filename}')

    field = item_value
    timeseries = df3[field]
    n_periods = 30

    start_test = (df3['Data'][0]).strftime("%d-%m-%Y")
    end_test = (df3['Data'][len(df3)-1]).strftime("%d-%m-%Y")

    print(f"Data inicio {start_test}")
    print(f"Data fim {end_test}")

    df3.Data = pd.to_datetime(df3['Data'])
    df3.set_index('Data', inplace=True)
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
        "assets/autoarima_01.png",
        dpi=300, bbox_inches='tight'
    )
    plt.show()

    # automodel = arimamodel(timeseries)
    data_order, automodel = model_auto_ARIMA(df3, field)

    plotarima(n_periods, automodel, df3, field)


def prediction_test(data_order, df2):

    global X
    X = df[field].values  # anos 1-1-2015 - 31-12-2019
    training_size = int(len(X) * 0.70)

    X2 = df2[field].values  # anos 1-1-2020 - 31-09-2020
    testing_size = int(len(X2))

    training, validate = X[:training_size], X2[:testing_size]

    print(training.shape, validate.shape)

    training_start = (df.index[0]).strftime("%d-%m-%Y")
    end_training = (df.index[len(training)-1]).strftime("%d-%m-%Y")

    testing_start_prediction = df2.index[0].strftime("%d-%m-%Y")
    end_testing_prediction = df2.index[len(X2)-1].strftime("%d-%m-%Y")

    # testing_start_prediction = datetime.datetime.strptime(
    #     '2020-01-01', '%Y-%m-%d')
    # end_testing_prediction = datetime.datetime.strptime(
    #     '2020-09-15', '%Y-%m-%d')

    model, predictions = train_test_predction(
        validate, training, data_order)

    print(
        f"Dados de treino: \ninicio: {training_start}, fim: {end_training}")
    print(
        f"Dados de validação: \ninicio: {testing_start_prediction}, fim: {end_testing_prediction}")

    res = model.fit(disp=False)
    print(res.summary())

    # evaluate forecasts
    rmse = sqrt(mean_squared_error(validate, predictions))
    print('Test RMSE: %.3f' % rmse)

    image_test_prediction('05', validate, predictions)


def predicton_SARIMAX(data_order, df2):

    y = df2[field].resample('D').mean()

    mode = SARIMAX(y, order=(data_order['order'][0], data_order['order'][1], data_order['order'][2]),
                   seasonal_order=(data_order['seasonal_order'][0],
                                   data_order['seasonal_order'][1],
                                   data_order['seasonal_order'][2],
                                   data_order['seasonal_order'][3]),
                   suppress_warnings=True,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    results = mode.fit()
    # print(results.summary().tables[1])
    print(results.summary())

    # pred = results.get_prediction(
    #     start=pd.to_datetime(df2.index[df2.shape[0]-1]), dynamic=False)
    pred = results.get_prediction(
        start=pd.to_datetime('2020-02-28'), dynamic=False)

    pred_ci = pred.conf_int()
    ax = y.plot(label='observed')
    pred.predicted_mean.plot(
        ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(18, 6))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature')

    plt.legend(("past", "forecast", "95% confidence interval"),
               loc="upper left")
    plt.show()
    # plt.tight_layout()
    plt.savefig(
        "assets/06.png"
    )

    print(f'############################## - {pred_ci}')

    # pic07(pred, y, results)


def pic07(pred, y, results):
    y_forecasted = pred.predicted_mean
    y_truth = y['2020-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(
        round(np.sqrt(mse), 2)))

    pred_uc = results.get_forecast(steps=60)
    pred_ci = pred_uc.conf_int()
    ax = y.plot(label='observed', figsize=(18, 6))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)

    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature')
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "assets/07.png",
        dpi=300, bbox_inches='tight'
    )

    res = (pred_ci['lower '+field] + pred_ci['upper '+field]) / 2
    print(res)
    pred_ci.head()

    # plt.show()

    # automodel = arimamodel(df2[field])

    # plotarima(30, df2[field], automodel)


def arimamodel(timeseries):
    automodel = auto_arima(timeseries,
                           start_p=1,
                           start_q=1,
                           test="adf",
                           seasonal=True,
                           trace=True)

    return automodel


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
    plt.savefig("assets/autoarima.png", dpi=300, bbox_inches='tight')
    plt.show()

    jsonMerged = {**json.loads(data_serie), **json.loads(data)}
    data = json.dumps(jsonMerged)
    print(data)

    return data
