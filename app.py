from modules import YOLOv8Trainer


def main():
    trainer = YOLOv8Trainer("config.yaml")

    model = trainer.configure_model()


if __name__ == "__main__":
    main()
