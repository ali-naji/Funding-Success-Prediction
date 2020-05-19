from funding_model.processing.preprocessor import Pipeline
from funding_model.config import config
from funding_model.manager import load_dataset, save_pipeline

pipeline = Pipeline(config.NUM_VARS, config.CAT_VARS,
                    config.STR_VARS, config.BOOL_VARS, config.DROP_VARS, config.TARGET, 200)

if __name__ == "__main__":
    data = load_dataset()
    pipeline.fit(data)
    pipeline.evaluate()
    save_pipeline(pipeline)
