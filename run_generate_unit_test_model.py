from trainers.trainer_simple import AutoGeneratorTrainer
import os

os.makedirs("./output/", exist_ok=True)

if __name__ == "__main__":
    AutoGeneratorTrainer().train().evaluate().save('./output/unit_test_model.pth')
