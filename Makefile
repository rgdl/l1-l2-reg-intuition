name := l1-l2-reg-intuition

build_base:
	docker build . -f base.Dockerfile -t ${name}-base

test: build_base
	docker build . -f test.Dockerfile -t ${name}-test
	docker run --rm -it ${name}-test
	docker image rm ${name}-test

build: build_base
	docker build . -f main.Dockerfile -t ${name}
	docker run --rm -p 8501:8501 ${name}
