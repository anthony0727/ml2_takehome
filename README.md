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

```docker-compose up -d```

url : aws.com

port

8080 airflow

8081-8082 ts

8500-8501 tfs

??? tensorboard

### Training

airflow triggering image

### Inference

[API table]

refer to [jupyter notebook] containing 

Followings are covered : 

* sample requests to each server
* benchmark with apache benchmark(ab)

## Architecture

TODO: Diagram

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

