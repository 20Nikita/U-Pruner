FROM paradigma_base:v0
RUN apt-get update
# Папка в которую вы попадаете, когда заходите в контейнер
WORKDIR /workspace/proj
COPY . . 
# Копируем туда же содержимое текущей папки 
RUN pip --no-cache-dir install -r /workspace/proj/requirements.txt # Тут ваши requirements
ENTRYPOINT ["python", "main.py"]
