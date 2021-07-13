import json
import os
import os.path


class KFPMetrics:
    def __init__(self, output_path: str):
        self.output_dir = os.path.dirname(output_path)
        self.output_path = output_path
        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics = []

    def add_metric(
        self, name: str, value: float, is_percentage: bool = False
    ) -> 'KFPMetrics':

        self.metrics.append(
            {
                'name': name,
                'numberValue': value,
                'format': 'PERCENTAGE' if is_percentage else 'RAW',
            }
        )
        return self

    def write(self):
        with open(self.output_path, 'w') as fp:
            json.dump({'metrics': self.metrics}, fp)
