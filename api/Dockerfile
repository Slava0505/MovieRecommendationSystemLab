FROM python:3.6-stretch
ENV PYTHONUNBUFFERED 1
COPY . /app
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 53
CMD [ "python3", "api.py" ]