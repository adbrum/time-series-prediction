import pandas as pd
from django.core.management.base import BaseCommand
from apps.predictions.models import SAGRAData
from sqlalchemy import create_engine
from django.conf import settings


class Command(BaseCommand):
    help = "A command to add data from an Excel file to the database"

    def handle(self, *args, **options):
        # return super().handle(*args, **options)
        df = pd.read_excel(
            'core/static/files/dadosSAGRA_Beja_16_09_2020.xls')
        # df.insert(2, 'created_on', '00/00/0000 00:00:00')
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
        },
            inplace=True, errors='raise')

        engine = create_engine('sqlite:///db.sqlite3')

        df.to_sql(SAGRAData._meta.db_table,
                  if_exists='replace', con=engine, index_label='id', index=True)
