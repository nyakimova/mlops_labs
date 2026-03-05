import argparse
from pathlib import Path

import pandas as pd


def resolve_csv_path(raw_dir: Path, name: str) -> Path:
    p = raw_dir / name
    if p.is_file():
        return p
    if p.is_dir():
        inner = p / name
        if inner.is_file():
            return inner
        csvs = list(p.glob("*.csv"))
        if len(csvs) == 1:
            return csvs[0]
        if len(csvs) > 1:
            raise ValueError(f"У папці {p} кілька CSV: {[c.name for c in csvs]}. Вкажи точний файл.")
    raise FileNotFoundError(f"Не знайдено {name} як файл або як папку з CSV у: {raw_dir}")


def main(args):
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_csv_path(raw_dir, "train.csv")
    features_path = resolve_csv_path(raw_dir, "features.csv")
    stores_path = resolve_csv_path(raw_dir, "stores.csv")

    train = pd.read_csv(train_path)
    features = pd.read_csv(features_path)
    stores = pd.read_csv(stores_path)

    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    data = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    data = data.merge(stores, on="Store", how="left")

    data["year"] = data["Date"].dt.year
    data["month"] = data["Date"].dt.month
    data["week"] = data["Date"].dt.isocalendar().week.astype(int)
    data = data.drop(columns=["Date"])

    data["IsHoliday"] = data["IsHoliday"].astype(str)
    if "Type" in data.columns:
        data["Type"] = data["Type"].astype(str)

    md_cols = [c for c in data.columns if c.lower().startswith("markdown")]
    if md_cols:
        data[md_cols] = data[md_cols].fillna(0)

    num_cols = data.select_dtypes(include="number").columns.tolist()
    if num_cols:
        data[num_cols] = data[num_cols].fillna(data[num_cols].median(numeric_only=True))

    out_path = out_dir / "train_prepared.parquet"
    data.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Shape:", data.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="mlops_lab_1/data/raw/walmart-recruiting-store-sales-forecasting",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="mlops_lab_1/data/processed",
    )
    args = parser.parse_args()
    main(args)