import os
import sys
import zipfile
import logging
import flogging

import retro.data

flogging.setup()
logger = logging.getLogger("import-roms")


def _check_zipfile(f, process_f):
    with zipfile.ZipFile(f) as zf:
        for entry in zf.infolist():
            _root, ext = os.path.splitext(entry.filename)  # noqa: PTH122
            with zf.open(entry) as innerf:
                if ext == ".zip":
                    _check_zipfile(innerf, process_f)
                else:
                    process_f(entry.filename, innerf)


def main():
    """Import ROMs from a directory into the retro data directory."""
    from retro.data import EMU_EXTENSIONS  # noqa: PLC0415

    # This avoids a bug when loading the emu_extensions.

    emu_extensions = {
        ".sfc": "Snes",
        ".md": "Genesis",
        ".sms": "Sms",
        ".gg": "GameGear",
        ".nes": "Nes",
        ".gba": "GbAdvance",
        ".gb": "GameBoy",
        ".gbc": "GbColor",
        ".a26": "Atari2600",
        ".pce": "PCEngine",
    }
    EMU_EXTENSIONS.update(emu_extensions)
    paths = sys.argv[1:] or [os.getcwd()]
    logger.info(f"Importing ROMs from: {paths}")
    logger.info("Fetching known hashes")
    known_hashes = retro.data.get_known_hashes()
    logger.info(f"Found {len(known_hashes)} known hashes")
    imported_games = 0

    def save_if_matches(filename, f):
        nonlocal imported_games
        try:
            data, hash = retro.data.groom_rom(filename, f)
        except (OSError, ValueError):
            logging.warning(f"Failed to process file: {filename}")
            return
        if hash in known_hashes:
            game, ext, curpath = known_hashes[hash]
            # print('Importing', game)
            rompath = os.path.join(curpath, game, f"rom{ext}")  # noqa: PTH118
            # print("ROM PATH", rompath)
            with open(rompath, "wb") as file:  # noqa: FURB103
                file.write(data)
            imported_games += 1
            logger.info(f"Imported game: {game}")
            logger.debug(f"to {rompath}")

    for path in paths:  # noqa: PLR1702
        logger.info(f"Processing path: {path}")
        for root, dirs, files in os.walk(path):
            for filename in files:
                logger.debug(f"Processing file: {root}/{filename}")
                filepath = os.path.join(root, filename)  # noqa: PTH118
                with open(filepath, "rb") as f:
                    _root, ext = os.path.splitext(filename)  # noqa: PTH122
                    if ext == ".zip":
                        try:
                            _check_zipfile(f, save_if_matches)
                        except (zipfile.BadZipFile, RuntimeError, OSError):
                            logger.warning(f"Failed to process zip file: {filepath}")
                    else:
                        save_if_matches(filename, f)
    logger.info(f"Total imported games: {imported_games}")


if __name__ == "__main__":
    sys.exit(main())
