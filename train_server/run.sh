        #train_server:
        #image: train_server:latest
        #build: ./train_server
        #tty: true
        #stdin_open: true
        #volumes:
        #    - /var/run/docker.sock:/var/run/docker.sock
        #    - ./models:/home/models
        #      #- ./train_server/app:/home/app

docker run -d --rm train_server -v /var/run/docker.sock:/var/run/docker.sock -v ../models:/home/model
