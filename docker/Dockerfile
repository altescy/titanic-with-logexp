FROM python:3.7

RUN pip install pipenv

WORKDIR /work

ADD . ./
RUN pipenv sync

CMD ["/bin/bash"]
