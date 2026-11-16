FROM apache/airflow:3.4.0

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==3.4.0" -r /requirements.txt