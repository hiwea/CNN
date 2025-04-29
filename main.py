# main.py
from config import Config
from trainer import Trainer


def main():
    config = Config()
    trainer = Trainer(config)
    x_train, y_train, x_test, y_test = trainer.load_and_preprocess_data()
    trainer.build_and_compile_model()
    trainer.train_model(x_train, y_train, x_test, y_test)
    trainer.plot_loss_curve()
    trainer.evaluate_model(x_test, y_test)
    trainer.run_experiment(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()