import logging, os

def setup_logger(name, output_dir, overwrite=True):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(message)s"
    )

    if not logger.handlers:

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            os.path.join(output_dir, f"{name}.log"),
            mode="w" if overwrite else "a",
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger

def log_training(logger, train_results, val_results, epoch):

    logger.info(f"Epoch {epoch:03d} | Train ktau τ: {train_results['ktau']:.3f} -- Train srho ρ:{train_results['srho']:.3f} -- Train mAP50: {train_results['map50']:.2f} -- Train mAP15: {train_results['map15']:.2f} -- Train Loss: {train_results['loss']:.6f} | Val ktau τ: {val_results['ktau']:.3f} -- Val srho ρ:{val_results['srho']:.3f} -- Val mAP50: {val_results['map50']:.2f} -- Val mAP15: {val_results['map15']:.2f} -- Val Loss: {val_results['loss']:.6f}")