import pandas as pd

file_dir="/lustre/grp/cyllab/share/evolution_prediction_dl"
tag="241030"
# tag="250516"

test_full = pd.read_csv(f"{file_dir}/data/processed/to{tag}/rbd/2023-10-01/TestFull.csv")
tasks = (
    test_full[["location", "t0", "n_bg_clusters"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
tasks.to_csv(f"{file_dir}/benchmark/MLR/res/task_list_TestFull_2023-10-01_to{tag}.csv", index=False)
print(f"Generated {len(tasks)} tasks")

å
