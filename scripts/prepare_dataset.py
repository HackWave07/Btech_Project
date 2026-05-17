import os

from src.config.config import Config
from src.data.dataset import prepare_dataset_csv

def main():
    print("Preparing dataset and estimating severity labels...")
    Config.setup_dirs()
    csv_path = os.path.join(Config.BASE_DIR, "dataset.csv")
    df = prepare_dataset_csv(Config.DATA_DIR, Config.ALLOWED_FOLDERS, csv_path)
    print(f"Dataset preparation complete. Saved to {csv_path}. Total samples: {len(df)}")
    print(df['severity'].value_counts())

if __name__ == "__main__":
    main()
