from src.utils import paths


def main(config_path=None, overrides=None):
    overrides = overrides or []
    print("config_path:", config_path)
    print("overrides:", overrides)
    print("DATA_DIR:", paths.DATA_DIR)
    print("EXPERIMENTS_DIR:", paths.EXPERIMENTS_DIR)
    return overrides


if __name__ == "__main__":
    main()
