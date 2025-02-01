mkdir weights
wget https://utexas.box.com/shared/static/3j3q9qsc1kovpwfxtnsful7pvdy234q6.tar -O ./weights/cpt_best_prob.pth.tar
gdown 'https://drive.google.com/uc?=id=1fehGhAGZbwwRsDYSH5atBn0YQ91SO6iu' -O ./weights/cpt_best_prob_no_transformer.pth.tar
# tar -xvf ./weights/weight.tar -C ./weights/