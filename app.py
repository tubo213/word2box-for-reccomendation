import streamlit as st
import pandas as pd
from pathlib import Path
import json
import torchtext
import random
from PIL import Image
import yaml
from word2box.src.language_modeling_with_boxes.models import Word2BoxConjunction
import torch


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_TEXT(path_dir: Path):
    vocab_stoi = load_json(path_dir / "vocab_stoi.json")
    vocab_freq = load_json(path_dir / "vocab_freq.json")

    TEXT = torchtext.data.Field()
    TEXT.stoi = vocab_stoi
    TEXT.freqs = vocab_freq
    TEXT.itos = [k for k, v in sorted(vocab_stoi.items(), key=lambda item: item[1])]

    # Since we won't train on <pad> and <eos>. These should not come in any sort of
    # subsampling and negative sampling part.
    TEXT.freqs["<pad>"] = 0
    TEXT.freqs["<unk>"] = 0

    return TEXT


def load_model(TEXT, config):
    model = Word2BoxConjunction(
        TEXT=TEXT,
        embedding_dim=config["embedding_dim"],
        batch_size=config["batch_size"],
        n_gram=config["n_gram"],
        intersection_temp=config["int_temp"],
        volume_temp=config["vol_temp"],
        box_type=config["box_type"],
    )

    # 作成したインスタンスに保存してあるパラメータを読み込む
    model.load_state_dict(torch.load(config['ckpt_path']))

    return model


@st.cache_data
def read_data(config):
    data_dir = Path(config["data_dir"])
    df = pd.read_csv(data_dir / "articles.csv")
    TEXT = load_TEXT(Path(config['vocab_dir']))
    model = load_model(TEXT, config['model_config'])
    return df, TEXT, model

def get_name2code(df, name_col, code_col):
    name2code = df.set_index(name_col)[code_col].drop_duplicates().to_dict()
    return name2code

def get_sub_data(df):
    prod_name2code = get_name2code(df, 'prod_name', 'product_code')
    product_type2_code = get_name2code(df, 'product_type_name', 'product_type_no')
    graphical_appearance2_code = get_name2code(df, 'graphical_appearance_name', 'graphical_appearance_no')
    color_group2_code = get_name2code(df, 'colour_group_name', 'colour_group_code')

    return prod_name2code, product_type2_code, graphical_appearance2_code, color_group2_code

def get_context(TEXT, product_code, product_type_no, graphical_appearance_no, colour_group_code):
    context = [
        f"product_code_{product_code}",
        f"product_type_no_{product_type_no}",
        f"graphical_appearance_no_{graphical_appearance_no}",
        f"colour_group_code_{colour_group_code}",
    ]
    # context = context[[product_code, product_type_no, graphical_appearance_no, colour_group_code]]
    context = torch.LongTensor([TEXT.stoi.get(w) for w in context if TEXT.stoi.get(w) is not None])

    return context

def select_conditional(unique_product_name, unique_product_type, unique_graphical_appearance, unique_color_group):
    st.sidebar.write("# Select conditional")
    product_name = st.sidebar.selectbox(
        "product name", unique_product_name
    )
    product_type_name = st.sidebar.selectbox(
        "product type", unique_product_type
    )
    graphical_appearance_name = st.sidebar.selectbox(
        "graphical appearance", unique_graphical_appearance
    )
    color_group_name = st.sidebar.selectbox(
        "color group", unique_color_group
    )

    return product_name, product_type_name, graphical_appearance_name, color_group_name


def main():
    config = load_yaml("./config.yaml")
    df, TEXT, model = read_data(config)

    st.title("Word2Box Demo")

    prod_name2code, product_type2_code, graphical_appearance2_code, color_group2_code = get_sub_data(df)
    product_name, product_type_name, graphical_appearance_name, color_group_name = select_conditional(
        [None] + list(prod_name2code.keys()),
        [None] + list(product_type2_code.keys()),
        [None] + list(graphical_appearance2_code.keys()),
        [None] + list(color_group2_code.keys())
    )

    product_code = prod_name2code.get(product_name)
    product_type_code = product_type2_code.get(product_type_name)
    graphical_appearance_code = graphical_appearance2_code.get(graphical_appearance_name)
    color_group_code = color_group2_code.get(color_group_name)

    unique_articles = [v for k, v in TEXT.stoi.items() if 'article_id' in k]
    id_to_word = {v: k for k, v in TEXT.stoi.items()}

    article = torch.LongTensor(unique_articles).unsqueeze(1)
    context = get_context(TEXT, product_code, product_type_code, graphical_appearance_code, color_group_code)
    context = context.expand(len(article), -1)
    mask = torch.ones(len(article), dtype=torch.bool)

    with torch.no_grad():
        model = model.eval().cuda()
        context = context.cuda()
        mask = mask.cuda()
        article = article.cuda()
        scores = model.forward(article, context, mask, train=False).flatten()

    topk_article_ids = article[scores.argsort(descending=True)[:config['topk']]].cpu().numpy()
    match_ids = [
        id_to_word[int(_id)].split('_')[-1] for _id in topk_article_ids
    ]
    match_ids = list(map(int, match_ids))

    display_cols = [
        'article_id',
        'prod_name',
        'product_type_name',
        'graphical_appearance_name',
        'colour_group_name'
    ]
    match_df = df.query(
        'article_id in @match_ids'
    ).reset_index(drop=True)

    st.write("# Matched articles")
    st.dataframe(match_df[display_cols])
    image_dir = Path(config['data_dir']) / 'images'

    num_columns = 4
    col = st.columns(num_columns)
    for i, row in match_df[display_cols].iterrows():
        j = i % num_columns
        with col[j]:
            article_id = str(row['article_id']).zfill(10)
            image_path = image_dir / article_id[:3] / f"{article_id}.jpg"

            st.write(f"### Rank {i}")
            if image_path.exists():
                img = Image.open(image_path)
                st.image(img, use_column_width=True)
            else:
                st.write("No image")
        if j == num_columns-1:
            col = st.columns(num_columns)


if __name__ == "__main__":
    main()
