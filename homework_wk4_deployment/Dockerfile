FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

COPY [ "starter_homework.py", "./" ]

COPY requirements.txt  /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV year='2021'
ENV month='4'

EXPOSE 9696

CMD ["python", "starter_homework.py"]
