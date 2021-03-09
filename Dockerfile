# build docker build -t tbalaz/spark:0.0.1 .
# run image: docker run -d -p 8888:8888 -p 4040:4040 -v C:\Users\werit\Documents\Latex\Intterview\Galytix\asignment\local_docker\work:/home/jovyan/work --name spark tbalaz/spark:0.0.1
# run command: docker exec -it spark /usr/local/spark/bin/spark-submit /home/jovyan/work/entry.py -s 'Sentence interesant for me'

FROM jupyter/pyspark-notebook

ARG DEV_HOME=/home/jovyan/work/
COPY entry.py ${DEV_HOME}
COPY manager.py ${DEV_HOME}
COPY services ${DEV_HOME}services
COPY Pipfile ${DEV_HOME}
COPY Pipfile.lock ${DEV_HOME}


RUN python -m install pipenv && cd ${DEV_HOME} && pipenv install 
