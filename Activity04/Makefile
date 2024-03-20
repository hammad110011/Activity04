setup:
	python -m venv venv
	. venv/bin/activate
	pip install -r requirements.txt

train:
	python model/train_model.py

run:
	python app/app.py

docker-build:
	docker build -t nlp_project .

docker-run:
	docker run -p 5000:5000 nlp_project
