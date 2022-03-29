
#!/bin/bash

codigo_http=$(curl --write-out %{http_code} --silent --output /dev/null sagra.ipbeja.pt:8000)

if [ $codigo_http -ne 302 ]; then

    echo "Houve um problema com o servidor, tentando reiniciá-lo  $(date +%F\ %T)" >> /home/aregis/scripts/logs/servidor.log

#    	nohup /home/aregis/time-series-prediction/venv/bin/python /home/aregis/time-series-prediction/manage.py runserver 0.0.0.0:8000 --insecure &
	source /home/aregis/time-series-prediction/venv/bin/activate
	cd /home/aregis/time-series-prediction/
     	nohup python manage.py runserver 0.0.0.0:8000 --insecure &

fi
