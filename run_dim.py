import os
import subprocess

def run_evaluation(seed, ssl_type, pool_type):
    seed = 7 

    model_path = f"C:/Users/user/Desktop/MSP/model/dim_ser/wavLM_adamW/{seed}"
    store_path = f"C:/Users/user/Desktop/MSP/result/dim_ser/wavLM_adamW/{seed}.txt"

    print(f"Model path: {model_path}")
    print(f"Store path: {store_path}")


    os.makedirs(os.path.dirname(store_path), exist_ok=True)

    try:
        subprocess.run([
            "python", "eval_dim_ser.py",
            "--ssl_type", ssl_type,
            "--pooling_type", pool_type,
            "--model_path", model_path,
            "--store_path", store_path
        ], check=True)

        print(f"Evaluation for seed {seed} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    ssl_type = "wavlm-large"
    pool_type = "AttentiveStatisticsPooling"
    seed = 7  

    run_evaluation(seed, ssl_type, pool_type)
