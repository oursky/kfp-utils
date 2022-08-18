import os
import os.path
from datetime import datetime
from typing import Dict


class HPTMetrics:
    def __init__(self, output_path: str):
        self.output_dir = os.path.dirname(output_path)
        self.output_path = output_path
        os.makedirs(self.output_dir, exist_ok=True)

    def write(self, data: Dict[str, float]):
        with open(self.output_path, 'a') as fp:
            for k, v in data.items():
                fp.write(
                    f'{datetime.utcnow().isoformat()}Z "{k}": "{v:.4f}"\n'
                )
