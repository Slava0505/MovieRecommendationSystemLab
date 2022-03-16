FROM python:3.6-stretch
ENV PYTHONUNBUFFERED 1
WORKDIR /app

COPY ./app /app
COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
EXPOSE 4000
CMD [ "python3", "api.py" ]