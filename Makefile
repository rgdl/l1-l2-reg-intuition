build:
	docker build . -t l1-l2-reg-intuition

run:
	docker run -p 8501:8501 l1-l2-reg-intuition
