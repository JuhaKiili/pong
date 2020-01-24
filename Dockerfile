FROM python:3.7
RUN pip install gym[atari]
RUN pip install ray
RUN pip install numpy
