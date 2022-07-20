FROM l1-l2-reg-intuition-base

COPY dev-requirements.txt .
RUN pip install -r dev-requirements.txt

COPY app /app
COPY test /test

CMD ["bash", "-c", "coverage run -m pytest test/test.py && coverage report"]

