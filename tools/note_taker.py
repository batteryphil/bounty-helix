import json
import os
from pathlib import Path
from typing import List, Dict

class NoteTaker:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)

    def save_note(self, label: str, content: str):
        file_path = self.data_dir / f"{label}.json"
        data = {"content": content}
        with open(file_path, "w") as f:
            json.dump(data, f)

    def get_notes(self, label: str) -> List[str]:
        file_path = self.data_dir / f"{label}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
            return [data["content"]]
        else:
            return []

if __name__ == "__main__":
    data_dir = Path("notes")
    note_taker = NoteTaker(data_dir)
    note_taker.save_note("test", "Hello, World!")
    print(note_taker.get_notes("test"))