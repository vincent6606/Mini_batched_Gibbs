!/bin/bash

python potts.py --states=10 --epoch=1000000 --size=20 ; python Gibbs_potts.py --states=10 --epoch=1000000 --size=20