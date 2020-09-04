docker run -d --rm -p 8501:8501 \
    -v "/home/anthony/repo/ml2_takehome/train_server/models/tensorflow/:/models/lenet5" \
    -e MODEL_NAME=lenet5 \
    -t tensorflow/serving
