import polars as pl
import yaml
from pathlib import Path

def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == "__main__":
    config = load_yaml("./config.yaml")
    data_dir = Path(config["data_dir"])
    transaction_path = data_dir / "transactions_train.csv"
    article_path = data_dir / "articles.csv"
    vocab_dir = Path(config['vocab_dir'])
    article_df = pl.read_csv(article_path)

    features = [
        "product_code",
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
    ]
    word_df = article_df.select(
        pl.col('article_id'),
        ("article_id_" + pl.col('article_id').cast(str)).alias("word")
    )
    for col in features:
        word_df = word_df.with_columns(
            word_df['word'] + " " +col + "_" + article_df[col].cast(str).alias("word")
        )

    pl.read_csv(transaction_path).join(
        word_df, on=['article_id'], how='left'
    ).groupby('customer_id').agg(
        pl.col('word').tail(config['max_seq']).str.concat(' ').alias('sentence')
    ).select('sentence').write_csv(vocab_dir / 'train.txt', has_header=False)
