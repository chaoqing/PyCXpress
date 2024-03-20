
install-deps:
	conda env create --file conda.yaml

install-deps-manually:
	conda create -n py38 python=3.8.10
	conda activate py38
	python3 -m pip install --upgrade pip
	python3 -m pip install pybind11 
	python3 -m pip install tensorflow==2.10.1
	conda env export | tee conda.yaml