# Art-Generator
To set up data:
-Download wikiart dataset and have the following directory structure: ./data/wikiart/..
-Run processdata.py

To run a model:
-Go into the gans or vaegan directory
-Run python cycle_gan.py --num_iters 70000 --sample_every 1000
	-can choose to also use cycle consistency by incorporating --use_cycle_consistency_loss
-It's easy to modify the network in models.py
-vaegan cycle_gan.py now is ready to run with all data by adding flag --run_all 
