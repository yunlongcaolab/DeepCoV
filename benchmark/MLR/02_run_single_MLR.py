import os
import sys
import argparse
import pandas as pd

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from jax import config, vmap
from jax.nn import softmax
import jax.numpy as jnp
import evofr as ef

config.update("jax_enable_x64", True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, required=True)
    parser.add_argument("--t0", type=str, required=True)
    parser.add_argument("--n_bg_clusters", type=int, required=True)
    parser.add_argument("--counts", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    
    args = parser.parse_args()

    model_type = {"MLR": ef.MultinomialLogisticRegression(tau=4.2)}
    svi = ef.InferFullRank(iters=50_000, lr=4e-3, num_samples=500)
    
    from data_reader import CountReader
    in_file = args.counts
    func = CountReader(in_file)

    background = func.query_background(
        loc=args.location,
        t0=args.t0,
        top_k=args.n_bg_clusters,
        n_bg_days=180,
        stride=1,
        shuffle=False
    )
    
    bg_names = func.index2name("sequence_names", background["indexes"].tolist())
    dates = pd.date_range(end=pd.to_datetime(args.t0) - pd.Timedelta(days=1), periods=180, freq="D")
    
    dfs = [pd.DataFrame({"date": dates, "location": args.location, "variant": name, "sequences": background["count"][i]}) 
           for i, name in enumerate(bg_names)]
    df_all = pd.concat(dfs, ignore_index=True)

    variant_data = ef.VariantFrequencies(df_all, pivot="r162")
    posterior = svi.fit(model_type["MLR"], variant_data)

    last_T = posterior.samples["freq"].shape[1]
    X = model_type["MLR"].make_ols_feature(start=last_T, stop=last_T + 32)
    logits = vmap(jnp.dot, in_axes=(None, 0))(X, jnp.array(posterior.samples["beta"]))
    posterior.samples["freq_forecast"] = softmax(logits, axis=-1)

    os.makedirs(args.outdir, exist_ok=True)
    freq_fr = pd.DataFrame(ef.get_freq(posterior.samples, variant_data, [0.95, 0.8, 0.5], name=args.location, forecast=True))
    out_path = f"{args.outdir}/freq_forecasted_{args.location}_{args.t0}.tsv"
    freq_fr.to_csv(out_path, sep="\t", index=False)
    print(f"Done: {args.location} {args.t0}")

if __name__ == "__main__":
    main()