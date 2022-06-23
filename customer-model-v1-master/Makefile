.PHONY: requirements environment rawdata processed join features labels validation staticdata all all_static

requirements:
	pip install --index-url https://build.sofi.com/artifactory/api/pypi/pypi/simple -r requirements.txt --use-deprecated=legacy-resolver

environment:
	chmod 700 env.sh
	./env.sh

rawdata:
	python src/data.py -r;\
    python src-transactional/data.py -r

processed:
	python src/data.py -p;\
	python src-transactional/data.py -p
    
join:
	python src/data.py -j;\
    python src-transactional/data.py -j

features:
	python src-transactional/data.py -f

labels:
	python src-transactional/data.py -l;\
	python src/data.py -l;\
    python src/data.py -c -af -nf;\
    
staticdata:
	python src/data.py -j;\
	python src-transactional/data.py -f;\
	python src/data.py -l;
	python src/data.py -c -af -nf;\

all: rawdata processed join features labels

all_static:
	python src/data.py -r;\
	python src/data.py -p;\
	python src/data.py -j;\
	python src-transactional/data.py -f;\
	python src/data.py -l;
	python src/data.py -c -af -nf;\