PWD=$(shell pwd)
PYTHON=pipenv run python
KAGGLE=pipenv run kaggle
JUPYTER=pipenv run jupyter
DATADIR=$(PWD)/data
DOCKER=docker
DOCKERFILE=$(PWD)/docker/Dockerfile
DOCKER_IMAGE=titanic
DOCKER_CONTAINER=titanic-container
DOCKER_PORT=8888


dataset:
	$(KAGGLE) competitions download -c titanic -p $(DATADIR)
	unzip -d$(DATADIR) $(DATADIR)/*.zip
	rm -f $(DATADIR)/*.zip

jupyter:
	$(JUPYTER) notebook

clean: clean-pyc

clean-pyc:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

docker: docker-build docker-run

docker-build:
	$(DOCKER) build -f $(DOCKERFILE) -t $(DOCKER_IMAGE) $(PWD)

docker-run:
	$(DOCKER) run -it -v $(PWD):/work -p $(DOCKER_PORT):$(DOCKER_PORT) --name $(DOCKER_CONTAINER) $(DOCKER_IMAGE)

docker-attach:
	$(DOCKER) attach $(DOCKER_CONTAINER)

docker-rm:
	$(DOCKER) rm $(DOCKER_CONTAINER)
