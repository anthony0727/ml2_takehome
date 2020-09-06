import os
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
schedule_interval = Variable.get("schedule_interval", deserialize_json=True)
dag = DAG(DAG_ID, default_args=default_args, schedule_interval=schedule_interval)

"""
sensors
"""

"""
model_sensor = SFTPSensor(
    task_id='mar_sensor',
    path=
    sftp_conn_id='sftp_default',
    poke_interval=5,
    dag=dag
)
"""

"""
tasks
"""

config = Variable.get('config', deserialize_json=True)
libtorch_config = config['libtorch']

train_libtorch = DockerOperator(
    task_id='train_libtorch',
    image='train_server',
    docker_url='unix://var/run/docker.sock',
    command='/home/app/lenet_libtorch/lenet5_libtorch {data_path} {model_path} {train_batch_sz} {test_batch_sz} {n_epochs} {lr}'.format(**libtorch_config),
    dag=dag
)

"""
pytorch_config = config['pytorch']
train_pytorch = DockerOperator(
    task_id='train_pytorch',
    image='train_server',
    docker_url='unix://var/run/docker.sock',
    dag=dag
)
"""

tensorflow_config = config['tensorflow']
train_tensorflow = DockerOperator(
    task_id='train_tensorflow',
    image='train_server',
    docker_url='unix://var/run/docker.sock',
    command='python3 /home/app/lenet5_tensorflow.py --epochs {epochs} --batch_size {batch_size} --lr {lr} --model_path {model_path}'.format(**tensorflow_config),
    dag=dag
)

"""
archive_model = DockerOperator(
    task_id='data',
    image='train_server',
    docker_url='unix://var/run/docker.sock',
    command='torch-model-archiver --model-name lenet5 --serialized-file model.pt --handler image_classifier',
    dag=dag
)
"""
train_libtorch >> train_tensorflow 
