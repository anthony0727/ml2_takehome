import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.operators.bash_operator import BashOperator

"""
DAG 
"""
default_args = {
    'owner': 'anthony',
    'depends_on_past': False,
    'start_date': datetime(2020, 9, 1),
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
common_args = dict(
    image='train_server',
    docker_url='unix://var/run/docker.sock',
    volumes=['/var/run/docker.sock:/var/run/docker.sock', '/home/ubuntu/ml2_takehome/models:/home/models'],
    xcom_push=True,
    xcom_all=True,
    tty=True,
    user='root',
    dag=dag
)

config = Variable.get('config', deserialize_json=True)

libtorch_config = config['libtorch']
train_libtorch = DockerOperator(
    task_id='train_libtorch',
    command='./main {data_path} {model_path} {train_batch_sz} {test_batch_sz} {n_epochs} {lr}'.format(**libtorch_config),
    working_dir='/home/app/lenet5_libtorch',
    **common_args
)

pytorch_config = config['pytorch']
train_pytorch = DockerOperator(
    task_id='train_pytorch',
    command='python3 lenet5_pytorch.py --data_path ./data --n_epochs {n_epochs} --train_batch_sz {train_batch_sz} --test_batch_sz {test_batch_sz} --lr {lr} --model_path {model_path}'.format(**pytorch_config),
    working_dir='/home/app',
    **common_args
)

tensorflow_config = config['tensorflow']
train_tensorflow = DockerOperator(
    task_id='train_tensorflow',
    command='python3 lenet5_tensorflow.py --epochs {epochs} --batch_size {batch_size} --lr {lr} --model_path {model_path}'.format(**tensorflow_config),
    working_dir='/home/app',
    **common_args
)


archive_libtorch = DockerOperator(
    task_id='archive_libtorch',
    command='torch-model-archiver --model-name lenet5 --serialized-file /home/ubuntu/ml2_takehome/models/libtorch/model.pt --handler image_classifier',
    **common_args
)

archive_pytorch = DockerOperator(
    task_id='archive_pytorch',
    command='torch-model-archiver --model-name lenet5 --serialized-file /home/ubuntu/ml2_takehome/models/libtorch/model.pt --handler image_classifier',
    **common_args
)

deploy_libtorch = BashOperator(
    task_id='deploy_libtorch',
    bash_command='echo libtorch',
    dag=dag
)

deploy_pytorch = BashOperator(
    task_id='deploy_pytorch',
    bash_command='echo pytorch',
    dag=dag
)

deploy_tensorflow = BashOperator(
    task_id='deploy_tensorflow',
    bash_command='echo tensorflow',
    dag=dag
)

train_libtorch >> train_pytorch >> train_tensorflow >> archive_libtorch >> archive_pytorch >> [deploy_libtorch, deploy_pytorch, deploy_tensorflow]
