
all: build

build_ext:
	python3 setup.py build_ext

model:
	git lfs install && git clone git@hf.co:moka-ai/m3e-base || echo 'clone m3e-base'

build: model
	docker build -t know-server -f docker/Dockerfile .

run:
	docker run --rm -p 8888:8888 know-server

FORCE:
