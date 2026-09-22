import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def download(output: Path, *, force: bool = False) -> Path:
    if output.exists() and not force:
        print(f"Dataset already exists: {output}")
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive = Path(temporary_directory) / "online-retail.zip"
        print(f"Downloading {DATASET_URL}")
        with urllib.request.urlopen(DATASET_URL, timeout=120) as response:
            with archive.open("wb") as destination:
                shutil.copyfileobj(response, destination)
        with zipfile.ZipFile(archive) as package:
            member = next(name for name in package.namelist() if name.lower().endswith(".xlsx"))
            with package.open(member) as source, output.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    print(f"Saved dataset to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the UCI Online Retail dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/online_retail/Online Retail.xlsx"),
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    download(arguments.output, force=arguments.force)


if __name__ == "__main__":
    main()
