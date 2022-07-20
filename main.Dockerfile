FROM l1-l2-reg-intuition-base

WORKDIR /app
COPY app /app

CMD ["streamlit", "run", "app.py"]

