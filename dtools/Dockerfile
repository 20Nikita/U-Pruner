FROM paradigma_base:v0
RUN apt-get update
WORKDIR /workspace/proj
COPY . .  
RUN pip --no-cache-dir install -r /workspace/proj/requirements.txt
ENTRYPOINT ["python", "main.py"]
