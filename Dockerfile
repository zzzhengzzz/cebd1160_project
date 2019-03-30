FROM ubuntu:16.04
MAINTAINER <Zheng Zheng >

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
pip install opencv-python
RUN pip3 install Numpy
RUN pip3 install pandas 
RUN pip3 install matplotlib
RUN pip3 install sklearn
WORKDIR /app
CMD ["python3", "/app/digits.py"]
