
all: build

build_ext:
	python3 setup.py build_ext

build:
	docker build -t know-server -f docker/Dockerfile .

run:
	docker run --rm -p 8888:8888 know-server

FORCE:
