from trainers.trainer_simple import AutoGeneratorTrainer

if __name__ == "__main__":
    AutoGeneratorTrainer().train().evaluate().save('./unit_test_model.pth')
