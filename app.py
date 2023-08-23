from modules import YOLOv8Trainer


def main():
    trainer = YOLOv8Trainer("config.yaml")
    model = trainer.configure_model()
    print(trainer)


if __name__ == "__main__":
    main()
