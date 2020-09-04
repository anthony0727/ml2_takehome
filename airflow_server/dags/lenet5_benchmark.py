import os
import random
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.docker_operator import DockerOperator

"""
DAG 
"""
default_args = {
    'owner': 'anthony',
    'depends_on_past': False,
    'start_date': datetime(2020, 6, 1),
    'email': ['anthony@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

DAG_ID = Path(__file__).stem
schedule_interval = Variable.get("schedule_interval", deserialize_json=True)['sise']
dag = DAG(DAG_ID, default_args=default_args, schedule_interval=schedule_interval)

"""
sensors
"""

# price_final_car2vec_2048x8_1024x8_dro_0.8_49
# sales_dur_final_car2vec_2048x8_1024x8_dro_0.8_49
# Depreciation_residual

price_pb_sensor = SFTPSensor(
    task_id='mar_sensor',
    path=pb_path('price'),
    sftp_conn_id='sftp_default',
    poke_interval=5,
    dag=dag
)

"""
tasks
"""

data = DockerOperator(
    task_id='data',
    image='train_server',
    docker_url='unix://var/run/docker.sock',
    dag=dag
)

train = DockerOperator(
    task_id='data',
    image='train_server',
    docker_url='unix://var/run/docker.sock',
    dag=dag
)

archive_model = DockerOperator(
    task_id='data',
    image='train_server',
    docker_url='unix://var/run/docker.sock',
    command='torch-model-archiver --model-name lenet5 --serialized-file model.pt --handler image_classifier',
    dag=dag
)