.PHONY: requirements environment rawdata processed join features labels all features_debug labels_debug all_debug query2feature train

requirements:
	pip install --index-url https://build.sofi.com/artifactory/api/pypi/pypi/simple -r requirements.txt

environment:
	./env.sh

rawdata:
	python main.py -r

processed:
	python main.py -p
    
join:
	python main.py -j

features:
	python main.py -f

labels:
	python main.py -l
    
features_debug:
	python main.py -f -d
    
labels_debug:
	python main.py -l -d
    
all_debug: 
	python main.py -d -r -p -j -f -l 

train:
	python main.py -t -c config.json
    
clean:
	python main.py --clean remove_non_current

all: rawdata processed join features labels

query2feature: rawdata processed join features