#Se utilizará Python
FROM python:3.9-slim

WORKDIR /app

COPY . /app

# Instalar requirements
RUN pip install -r requirements.txt

# Puerto 5000
EXPOSE 5000

ENV NAME World

CMD ["python", "movie_recomendations.py"]
