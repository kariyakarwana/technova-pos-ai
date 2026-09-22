import argparse
import csv
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_COLUMNS = [
    "invoice_no",
    "product_id",
    "description",
    "quantity",
    "invoice_date",
    "unit_price",
    "customer_id",
    "country",
]

NAMESPACE = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
COLUMN_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}


def _shared_strings(package: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in package.namelist():
        return []
    values: list[str] = []
    with package.open(path) as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag == f"{NAMESPACE}si":
                values.append("".join(node.text or "" for node in element.iter(f"{NAMESPACE}t")))
                element.clear()
    return values


def _value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find(f"{NAMESPACE}v")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{NAMESPACE}t"))
    if value_node is None or value_node.text is None:
        return ""
    value = value_node.text
    if cell_type == "s":
        return shared_strings[int(value)]
    return value


def _excel_datetime(serial: str) -> str:
    value = float(serial)
    converted = datetime(1899, 12, 30) + timedelta(days=value)
    return converted.isoformat()


def preprocess(input_path: Path, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with zipfile.ZipFile(input_path) as package:
        strings = _shared_strings(package)
        with (
            package.open("xl/worksheets/sheet1.xml") as worksheet,
            output_path.open("w", encoding="utf-8", newline="") as stream,
        ):
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for _, row in ET.iterparse(worksheet, events=("end",)):
                if row.tag != f"{NAMESPACE}row":
                    continue
                if int(row.attrib.get("r", "0")) == 1:
                    row.clear()
                    continue
                values = [""] * 8
                for cell in row.findall(f"{NAMESPACE}c"):
                    column = "".join(
                        character for character in cell.attrib["r"] if character.isalpha()
                    )
                    if column in COLUMN_INDEX:
                        values[COLUMN_INDEX[column]] = _value(cell, strings)
                (
                    invoice,
                    product,
                    description,
                    quantity_text,
                    date_text,
                    price_text,
                    customer,
                    country,
                ) = values
                try:
                    quantity = float(quantity_text)
                    unit_price = float(price_text)
                except ValueError:
                    skipped += 1
                    row.clear()
                    continue
                if (
                    not invoice
                    or invoice.upper().startswith("C")
                    or not product
                    or quantity <= 0
                    or unit_price <= 0
                    or not date_text
                ):
                    skipped += 1
                    row.clear()
                    continue
                writer.writerow(
                    dict(
                        zip(
                            OUTPUT_COLUMNS,
                            [
                                invoice,
                                product,
                                description,
                                quantity,
                                _excel_datetime(date_text),
                                unit_price,
                                customer,
                                country,
                            ],
                            strict=True,
                        )
                    )
                )
                written += 1
                row.clear()
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream and normalize the UCI retail workbook.")
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/online_retail/Online Retail.xlsx")
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/online_retail.csv"))
    arguments = parser.parse_args()
    written, skipped = preprocess(arguments.input, arguments.output)
    print(f"Wrote {written:,} valid transactions to {arguments.output}")
    print(f"Skipped {skipped:,} cancelled or invalid transactions")


if __name__ == "__main__":
    main()
