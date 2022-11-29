FROM python:3.8-slim-buster

WORKDIR /src

COPY src/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
COPY src/ .
ENTRYPOINT ["python"]
CMD ["main.py"]