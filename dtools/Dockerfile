FROM paradigma/base:0.2p
RUN apt-get update
COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt
WORKDIR /workspace/proj
COPY . .
ENTRYPOINT ["python", "-m","code"]
