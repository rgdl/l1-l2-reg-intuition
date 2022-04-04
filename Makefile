name := l1-l2-reg-intuition

tests:
	coverage run -m pytest test/test.py
	coverage report

build:
	docker build . -t ${name}

run:
	docker run -p 8501:8501 ${name}
