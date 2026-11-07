
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import io
import time
import zipfile


START_DATE = date(2025, 8, 12)
END_DATE = date(2026, 8, 11)

OUTPUT_DIR = Path(
    "data/raw/options/historical_2025-08-12_to_2026-08-11"
)

BASE_URL = (
    "https://archives.nseindia.com/content/fo/"
    "BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/zip,application/octet-stream,*/*",
}


def download_file(trading_date):
    date_code = trading_date.strftime("%Y%m%d")

    filename = f"nifty_fo_bhavcopy_{trading_date}.zip"
    output_path = OUTPUT_DIR / filename

    if output_path.exists() and zipfile.is_zipfile(output_path):
        return "EXISTS"

    url = BASE_URL.format(date=date_code)
    request = Request(url, headers=HEADERS)

    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()

        # Confirm the server returned a valid ZIP file.
        if not zipfile.is_zipfile(io.BytesIO(content)):
            return "INVALID"

        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_files = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                return "INVALID"

        output_path.write_bytes(content)

        return "DOWNLOADED"

    except HTTPError as error:
        if error.code == 404:
            return "NOT_AVAILABLE"

        if error.code in (403, 429):
            return f"STOP_HTTP_{error.code}"

        return f"HTTP_ERROR_{error.code}"

    except (URLError, TimeoutError, OSError) as error:
        return f"NETWORK_ERROR: {error}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current_date = START_DATE

    downloaded = 0
    existing = 0
    unavailable = 0
    invalid = 0
    errors = 0
    attempted = 0

    total_days = (END_DATE - START_DATE).days + 1

    print("NSE NIFTY OPTIONS HISTORICAL DOWNLOADER")
    print("---------------------------------------")
    print("Start date:", START_DATE)
    print("End date  :", END_DATE)
    print("Calendar days to check:", total_days)
    print("Output folder:", OUTPUT_DIR)
    print("\nStarting download...\n")

    while current_date <= END_DATE:
        attempted += 1

        result = download_file(current_date)

        if result == "DOWNLOADED":
            downloaded += 1
            print(f"[DOWNLOADED] {current_date}")

        elif result == "EXISTS":
            existing += 1

        elif result == "NOT_AVAILABLE":
            unavailable += 1

        elif result == "INVALID":
            invalid += 1
            print(f"[INVALID FILE] {current_date}")

        elif result.startswith("STOP_HTTP_"):
            errors += 1
            print(f"[STOPPED] {current_date}: {result}")
            print("Server blocked or rate-limited the request.")
            break

        else:
            errors += 1
            print(f"[ERROR] {current_date}: {result}")

        current_date += timedelta(days=1)
        time.sleep(0.25)

    print("\nDOWNLOAD SUMMARY")
    print("----------------")
    print("Dates attempted :", attempted)
    print("Downloaded      :", downloaded)
    print("Already existed :", existing)
    print("Not available   :", unavailable)
    print("Invalid files   :", invalid)
    print("Errors          :", errors)

    print("\nDownload folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
