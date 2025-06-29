# 📦 src/api/main.py
from src import data_processing, train, predict

def run_all():
    print("🚀 Running data processing...")
    data_processing.main()

    print("🚀 Running model training...")
    train.main()

    print("🚀 Running prediction...")
    predict.main()

if __name__ == "__main__":
    run_all()
