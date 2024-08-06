conda create --name mutarget python=3.10
conda activate mutarget
conda install pytorch==1.13 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install conda-forge::pandas
conda install conda-forge::pyyaml
pip install python-box[all]~=7.0 --upgrade
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install pytest-shutil
pip install peft==0.11.0
pip install transformers==4.38.2
pip install scikit-learn
