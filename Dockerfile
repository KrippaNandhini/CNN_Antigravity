FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create uploads directory and set permissions for potentially non-root user
RUN mkdir -p uploads && chmod 777 uploads

EXPOSE 7860

CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
