cd /lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/growth_advantage
awk 'FNR==1 && NR!=1 {next} {print}' results/rbd_test_GA_trunk*.csv > rbd_test_XBBera_ga.csv

# check 
wc -l /lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/rbd/2022-09-01/TestFull.csv
wc -l rbd_test_XBBera_ga.csv


### spike
cd /lustre/grp/cyllab/yangsj/evo_pred/1article/benchmark/growth_advantage
awk 'FNR==1 && NR!=1 {next} {print}' results/spike_test_GA_trunk*.csv > spike_test_JN1era_ga.csv

wc -l /lustre/grp/cyllab/yangsj/evo_pred/1article/data/processed/to241030/spike/2023-10-01/TestFull.csv
wc -l spike_test_JN1era_ga.csv


