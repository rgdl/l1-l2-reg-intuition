name := l1-l2-reg-intuition

build_base:
	docker build . -f docker/base.Dockerfile -t ${name}-base

test: build_base
	docker build . -f docker/test.Dockerfile -t ${name}-test
	docker run --rm -it ${name}-test

build: build_base
	docker build . -f docker/main.Dockerfile -t ${name}
	docker run --rm -it -p 8501:8501 ${name}
