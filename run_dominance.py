# -*- coding: UTF-8 -*-
import os
import subprocess
import argparse

def run_evaluation(seed, ssl_type, pool_type):
    model_path = f"C:/Users/user/Desktop/MSP/model/dim_dom_ser/wavLM_adamW/{seed}"
    store_path = f"C:/Users/user/Desktop/MSP/result/dim_dom_ser/wavLM_adamW/{seed}.txt"

    os.makedirs(os.path.dirname(store_path), exist_ok=True)

    try:
        subprocess.run([
            "python", "eval_dom_dim_ser.py",
            "--ssl_type", ssl_type,
            "--pooling_type", pool_type,
            "--model_path", model_path,
            "--store_path", store_path
        ], check=True)

        print(f"Evaluation for seed {seed} completed successfully. Results saved to {store_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    ssl_type = "wavlm-large"
    pool_type = "AttentiveStatisticsPooling"
    seed = 7  
    run_evaluation(seed, ssl_type, pool_type)
