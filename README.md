# ml2_takehome

## Objective

Implement Lenet5 with Libtorch

Additionally, I'm intending to advance the project to some extent

Which inlcudes:

* Benchmark Performance
  * training time: Libtorch vs Pytorch vs Tensorflow
  * inference TPS: pytroch/torchserve(ts) vs tensorflow/serving(tfs)

* Orchestration
  * automate ml lifecycle with Apache Airflow

* Dockerize
  * all logics are modularized with docker container

* If time allows, gather all models around and monitor them with tensorboard

## Usage

### build & run

```docker-compose up -d```

url : http://ec2-3-35-42-27.ap-northeast-2.compute.amazonaws.com

|                    | port                                     |
|--------------------|------------------------------------------|
| airflow            | 8080: admin                              |
| torchserve         | 8081: Management API 8082: Inference API |
| tensorflow/serving | 8500: gRPC 8501: REST API                |

### Training

![](img/airflow.png)

click play button

### Inference

tensorflow/serving

POST http://host:port/v1/models/${MODEL_NAME}[/versions/${VERSION}|/labels/${LABEL}]:predict

torchserve

curl http://127.0.0.1:8080/predictions/densenet161 -T kitten.jpg

refer to [jupyter notebook](https://colab.research.google.com/github/anthony0727/ml2_takehome/blob/master/notebook/lenet5_benchmark.ipynb) 

Followings are covered : 

* sample requests to each server
* benchmark with apache benchmark(ab)

## Architecture

![](img/archi.png)

## Modeling

All models are to be implemented in same conditions

Layers

Optimizer : Adam

* learning rate: 0.01
* beta1: 0.9
* beta2: 0.999
* epsilon:

## Code Review

Since this project is mainly focused on implementation of Lenet5 with Libtorch,

Libtorch C++ code is reviewed [here]

## Dev Note

having trouble with loading TorchScript generated model to torchserve

```2020-09-04 12:36:07,574 [INFO ] W-9003-lenet5_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - torch.nn.modules.module.ModuleAttributeError: 'RecursiveScriptModule' object has no attribute 'forward```

why torchserve cannot recognize model?

looking into load_model logic in torchserve, still can't figure out why

## Reference

docker.sock
